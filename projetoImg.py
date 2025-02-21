#############################
# IMPORTS
#############################
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


#############################
# 1) FUNÇÃO PARA CARREGAR IMAGENS E RÓTULOS
#############################
def load_images_and_labels(base_path, class_names, img_size=(224, 224)):
    """
    Carrega imagens de subpastas definidas em 'class_names', redimensiona para 'img_size'
    e retorna arrays NumPy contendo as imagens e seus rótulos.

    Parâmetros:
        base_path (str): Caminho base onde as subpastas estão localizadas (ex.: 'dataset').
        class_names (list): Lista com os nomes das classes/subpastas (ex.: ['covid', 'normal']).
        img_size (tuple): Tamanho (width, height) para redimensionar as imagens.

    Retorna:
        X (np.ndarray): Array de imagens com shape (N, height, width, 3).
        y (np.ndarray): Array de rótulos com shape (N,).
    """
    images, labels = [], []

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(base_path, class_name)
        image_files = glob.glob(os.path.join(class_folder, '*.*'))

        for file_path in image_files:
            try:
                with Image.open(file_path) as img:
                    img_rgb = img.convert('RGB')
                    img_resized = img_rgb.resize(img_size)
                    images.append(np.array(img_resized, dtype=np.uint8))
                    labels.append(label)
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")
                continue

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y


#############################
# 2) CONFIGURAÇÃO DO DATASET
#############################
BASE_PATH = "train"  # Ajuste conforme a localização do seu dataset
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]
IMG_SIZE = (224, 224)

# Carregar as imagens e os rótulos
X, y = load_images_and_labels(BASE_PATH, CLASS_NAMES, img_size=IMG_SIZE)
print("Total de imagens carregadas:", len(X))
print("Shape de X:", X.shape)
print("Shape de y:", y.shape)

#############################
# 3) DIVISÃO EM TREINO E TESTE
#############################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


#############################
# 4) ABORDAGEM CLÁSSICA: EXTRAÇÃO DE HOG + SVM
#############################
def extract_hog_features(images):
    """
    Extrai características HOG para cada imagem em uma coleção.

    Parâmetros:
        images (np.ndarray): Array de imagens no formato (N, H, W, 3) em RGB.

    Retorna:
        np.ndarray: Array com os vetores HOG para cada imagem (N, num_features).
    """
    hog_features = []

    for idx in range(images.shape[0]):
        # Converter para escala de cinza utilizando uma combinação linear dos canais RGB
        img_gray = np.dot(images[idx][..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        features = hog(
            img_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True
        )
        hog_features.append(features)

    return np.array(hog_features, dtype=np.float32)


# Extração das características HOG para treino e teste
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Treinamento do classificador SVM com kernel linear
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_hog, y_train)

# Avaliação com o SVM
y_pred_svm = svm_classifier.predict(X_test_hog)
print("Relatório de Classificação (SVM + HOG):")
print(classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES))

#############################
# 5) ABORDAGEM COM CNN: FINE-TUNING DO RESNET18 COM PYTORCH
#############################
# Definir as transformações para as imagens
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def apply_transform(image_array):
    """
    Aplica as transformações definidas para converter uma imagem NumPy em um tensor normalizado.

    Parâmetros:
        image_array (np.ndarray): Imagem em formato NumPy.

    Retorna:
        torch.Tensor: Imagem transformada.
    """
    return data_transform(Image.fromarray(image_array))


# Aplicar as transformações em lote para os conjuntos de treino e teste
X_train_tensor = torch.stack([apply_transform(img) for img in X_train])
X_test_tensor = torch.stack([apply_transform(img) for img in X_test])

# Converter os rótulos para tensores do PyTorch
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

# Carregar o modelo ResNet18 pré-treinado e congelar os pesos
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

# Ajustar a camada final para o número de classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_NAMES))

# Configurar dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)


def train_one_epoch(model, inputs, labels, optimizer, criterion, batch_size=16):
    """
    Treina o modelo por uma época.

    Parâmetros:
        model (torch.nn.Module): O modelo a ser treinado.
        inputs (torch.Tensor): Tensores de entrada.
        labels (torch.Tensor): Rótulos correspondentes.
        optimizer: Otimizador.
        criterion: Função de perda.
        batch_size (int): Tamanho do lote.

    Retorna:
        float: Perda média da época.
    """
    model.train()
    permutation = torch.randperm(inputs.size(0))
    running_loss = 0.0

    for start_idx in range(0, inputs.size(0), batch_size):
        batch_indices = permutation[start_idx:start_idx + batch_size]
        batch_inputs = inputs[batch_indices].to(device)
        batch_labels = labels[batch_indices].to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)

    epoch_loss = running_loss / inputs.size(0)
    return epoch_loss


# Treinamento por 5 épocas
num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = train_one_epoch(model, X_train_tensor, y_train_tensor, optimizer, criterion, batch_size=16)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

#############################
# 6) PLOTAR MATRIZ DE CONFUSÃO (SVM + HOG)
#############################
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Matriz de Confusão - SVM + HOG")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.tight_layout()
plt.show()
