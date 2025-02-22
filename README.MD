# Projeto Final - Processamento de Imagens

## Jeffrey de Souza

### 1. Descrição Geral do Projeto

Este projeto tem como objetivo aplicar técnicas de classificação de imagens utilizando duas abordagens distintas:

- **Abordagem Clássica:**  
  Utiliza a extração de características com **Histogram of Oriented Gradients (HOG)** combinada com um classificador tradicional **Support Vector Machine (SVM)**.  
  O descritor HOG capta a distribuição de gradientes e orientações locais, permitindo a distinção de texturas e contornos.

- **Abordagem com Redes Neurais Convolucionais (CNNs):**  
  Realiza o fine-tuning de um modelo pré-treinado, neste caso, o **ResNet18** do PyTorch, adaptando-o para classificar imagens do dataset escolhido.

O dataset utilizado é o [Coffee Bean Dataset (Resized 224x224)](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224), que contém imagens de grãos de café com diferentes características (por exemplo, "Dark", "Green", "Light", "Medium"). Esse dataset possibilita uma análise significativa, pois possui várias amostras por classe.

### 2. Equipe

- Jeffrey de Souza

### 3. Descritores Implementados

- **HOG (Histogram of Oriented Gradients):**  
  Extrai informações sobre os contornos e texturas das imagens, através da distribuição de gradientes.

- **CNN (ResNet18 Fine-Tuning):**  
  Utiliza um modelo pré-treinado para realizar a classificação das imagens, com ajuste fino na camada final para adaptar o modelo ao número de classes do dataset.

### 4. Repositório do Projeto

[Link para o repositório do projeto](https://drive.google.com/drive/folders/1JIu8gPZLnwPyzEM8-YgomC-jJLyRIS5o?usp=drive_link)

### 5. Classificadores e Acurácia

#### Abordagem Clássica (HOG + SVM)
- **Acurácia Obtida:** 91.2%
- **Outras Métricas:**  
  - Precision: 90.5%  
  - Recall: 92.0%  
  - F1-Score: 91.2%

#### Abordagem com CNN (ResNet18 Fine-Tuning)
- **Acurácia Obtida:** 94.7%
- **Outras Métricas:**  
  - Precision: 94.0%  
  - Recall: 95.0%  
  - F1-Score: 94.5%

### 6. Instruções de Uso

#### Pré-requisitos:
- **Ambiente:** Google Colab ou ambiente local com Python 3.x.
- **Bibliotecas Necessárias:**  
  - NumPy  
  - Matplotlib  
  - Seaborn  
  - Pillow (PIL)  
  - scikit-image  
  - scikit-learn  
  - PyTorch  
  - torchvision
