# portfolio_redes_neurais
Projetos de Data Science e Analytics
# 🩺 Detecção de Neoplasias Mamárias com Redes Neurais Convolucionais

> TCC — MBA em Data Science & Analytics | USP/ESALQ 2025  
> Autor: André Luiz Magalhães de Oliveira

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Concluído-brightgreen?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## 📌 Visão Geral

Este projeto investiga o uso de **Redes Neurais Convolucionais (ResNet50)** combinadas com técnicas de balanceamento e redução de dimensionalidade para **classificação de neoplasias mamárias** em imagens de mamografia.

O objetivo não é substituir o médico, mas **auxiliar o diagnóstico clínico**, melhorando a detecção precoce do câncer de mama — que representa mais de 73.000 novos casos estimados no Brasil até 2025.

---

## 🗂️ Estrutura do Repositório

```
breast-cancer-cnn/
│
├── notebook/
│   └── breast_cancer_classification.ipynb   # Notebook principal com todo o pipeline
│
├── data/
│   └── README_data.md                       # Instruções para download do dataset CBIS-DDSM
│
├── results/
│   ├── confusion_matrices/                  # Matrizes de confusão dos modelos
│   └── roc_curves/                          # Curvas ROC e Precision-Recall
│
└── README.md
```

---

## 📊 Dataset

**CBIS-DDSM** — *Curated Breast Imaging Subset of Digital Database for Screening Mammography*  
Fonte: [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

- Imagens de mamografia em formato DICOM
- Arquivos `.csv` com metadados para treino e teste
- Parâmetros utilizados: **densidade da mama**, **presença/forma de nódulos** e **tipo de microcalcificações**
- Divisão: **75% treino / 25% teste**

---

## ⚙️ Metodologia

O pipeline foi desenvolvido em etapas progressivas, com foco em resolver o **desbalanceamento de classes** (mais casos negativos do que positivos no dataset):

### Etapa 1 — ResNet50 (baseline)
Classificação direta das imagens de mamografia usando a rede pré-treinada ResNet50.

### Etapa 2 — SMOTE + PCA + Random Forest + ResNet50
- **SMOTE**: cria amostras sintéticas da classe minoritária para balancear o dataset  
- **PCA**: reduz a dimensionalidade, atenuando o efeito de bordas e texturas indesejadas  
- **Random Forest**: conjunto de árvores de decisão para reduzir overfitting e ruído

### Etapa 3 — SMOTEENN + Random Forest + ResNet50
- **SMOTEENN**: combina SMOTE com ENN (*Edited Nearest Neighbor*) para limpeza de ruído nas fronteiras de classe

---

## 📈 Resultados

| Modelo | Acurácia | Precisão (cl.1) | Recall (cl.1) | AUC-ROC |
|---|---|---|---|---|
| ResNet50 (baseline) | 66% | 66% | 47% | — |
| SMOTE + PCA + RF + ResNet50 | **70%** | **61%** | **67%** | **0.77** |
| SMOTEENN + RF + ResNet50 | 66% | 53% | 81% | 0.68 |

> **Melhor modelo geral:** `SMOTE + PCA + Random Forest + ResNet50`  
> AUC = 0.77 (classificado como "Bom" segundo escala clínica) e AP = 0.67

**Nota sobre a escolha:** apesar do SMOTEENN apresentar recall de classe 1 mais alto (81%), a remoção de dados pelo ENN comprometeu acurácia e precisão. O modelo SMOTE+PCA+RF mostrou melhor equilíbrio entre todas as métricas, sendo mais confiável para aplicação clínica.

---

## 🧰 Tecnologias Utilizadas

| Categoria | Biblioteca |
|---|---|
| Deep Learning | TensorFlow / Keras (ResNet50) |
| Balanceamento | imbalanced-learn (SMOTE, SMOTEENN) |
| Redução dimensional | scikit-learn (PCA) |
| Classificação | scikit-learn (Random Forest) |
| Análise | NumPy, Pandas |
| Visualização | Matplotlib, Seaborn |

---

## 🚀 Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/breast-cancer-cnn.git
cd breast-cancer-cnn

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o dataset (instruções em data/README_data.md)

# 4. Execute o notebook
jupyter notebook notebook/breast_cancer_classification.ipynb
```

---

## 🔭 Trabalhos Futuros

- Aplicar arquiteturas mais especializadas: **Vision Transformers (ViT)** e **Swin Transformers**
- Ajuste fino de hiperparâmetros para aumentar recall de classe 1 para a faixa ideal (80–90%)
- Avaliação com curvas ROC por subgrupo (tipo de tumor, densidade da mama)
- Integração com sistema de apoio à decisão clínica

---

## 📄 Publicação

> Oliveira, A. L. M.; Bampi, H. *Utilização de redes neurais convolucionais para identificação de neoplasias mamárias.* Trabalho de Conclusão de Curso — Especialização em Data Science & Analytics, USP/ESALQ, 2025.

---

## 📬 Contato

**André Luiz Magalhães de Oliveira**  
Graduação em Física Médica — USP Ribeirão Preto  
📧 andreluizmoliveira7@gmail.com  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/seu-perfil)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat-square&logo=github)](https://github.com/seu-usuario)
