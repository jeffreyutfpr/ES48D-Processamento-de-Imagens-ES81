# Projeto de Classificação de Imagens

## Jeffrey de Souza

## Descrição dos Descritores Implementados
- **Nome do Descritor:** HOG (Histogram of Oriented Gradients)
- **Características Extraídas:** O descritor HOG extrai informações sobre a distribuição de gradientes e orientações locais das imagens, permitindo capturar detalhes de contornos e texturas que auxiliam na distinção entre as classes (por exemplo, "covid" vs "normal").

## Repositório do Projeto
[Link do repositório do projeto](https://drive.google.com/drive/folders/1JIu8gPZLnwPyzEM8-YgomC-jJLyRIS5o?usp=drive_link)

## Classificador e Acurácia
- **Classificador Clássico:** SVM com HOG  
  - **Acurácia Obtida:** *Informe a acurácia, por exemplo, 92%*
- **Classificador com CNN:** ResNet18 Fine-Tuning com PyTorch  
  - **Acurácia Obtida:** *Informe a acurácia, por exemplo, 95%*

## Instruções de Uso (Opcional)
- **Pré-requisitos:**  
  - Python 3.x  
  - Bibliotecas: NumPy, Matplotlib, Seaborn, PIL, scikit-image, scikit-learn, PyTorch, torchvision
- **Como Executar:**  
  1. Clone o repositório: `git clone URL_DO_REPOSITORIO`
  2. Instale as dependências: `pip install -r requirements.txt`
  3. Execute o script principal: `python nome_do_script.py`
- **Observações:**  
  - Ajuste os caminhos dos diretórios conforme necessário.
  - O treinamento com a CNN pode demorar se não houver uma GPU disponível.
