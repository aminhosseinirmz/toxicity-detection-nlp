# Toxic Comment Classification — Deep Learning Survey & Implementation

## 📌 Overview
This project presents a comparative study of deep learning architectures for toxic comment detection and sentence semantic classification. The goal is to evaluate how different models capture both local textual features and long-range semantic dependencies in a multi-label classification setting.

Toxicity detection is a critical problem in modern online platforms, where automated systems are required to identify harmful content such as insults, threats, and hate speech.

---

## 🎯 Objectives
- Implement and compare multiple deep learning models for toxicity detection  
- Analyze strengths and weaknesses of each architecture  
- Evaluate performance using standard classification metrics  
- Benchmark models against a strong baseline (BERT)  

---

## 🗂 Dataset
- Wikipedia Toxic Comments Dataset (Jigsaw)
- Multi-label classification task with the following labels:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate  

This dataset reflects real-world toxic behavior in online discussions.

Download the dataset from:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

---

## 🏗 Models Implemented

### 🔹 Baseline
- MLP (Multi-Layer Perceptron)

### 🔹 Deep Learning Models
- CNN – captures local patterns (n-grams)
- C-LSTM – combines CNN + LSTM for spatial + sequential modeling
- CNN-BiGRU – hybrid model for feature extraction + temporal dependencies
- MCBiGRU – multi-channel CNN + BiGRU for richer representations

### 🔹 Benchmark Model
- BERT (bert-base-cased)

---

## ⚙️ Methodology

### Preprocessing
- Tokenization using SpaCy
- Vocabulary building with PyTorch
- Use of GloVe embeddings (100d, 300d)

### Training Setup
- Optimizers: Adam / Adadelta / AdamW  
- Sequence lengths: 25 and 50 tokens  
- Epochs: up to 50 (BERT: 2 epochs)  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUROC (primary metric)

---

## 📊 Results Summary

| Model        | Validation AUROC | Notes |
|-------------|----------------|------|
| **BERT** | **0.99** | Best performance, fastest convergence |
| MCBiGRU | ~0.978 | Strong hybrid model |
| CNN-BiGRU | ~0.971 | Competitive, stable |
| CNN | ~0.97 | Good baseline DL model |
| C-LSTM | ~0.967 | Strong semantic modeling |
| MLP | ~0.69 | Weak baseline |

👉 BERT clearly outperforms all models with minimal training  
👉 Hybrid architectures (CNN + RNN) outperform simpler models  

---

## 📈 Key Insights
- Hybrid models outperform single architectures  
- BERT dominates due to pretrained representations  
- Sequence modeling is crucial for text classification  
- MLP is not suitable for sequential data  

---

## ⚠️ Limitations
- CNN struggles with long-range dependencies  
- GRU-based models may miss complex temporal patterns  
- MLP cannot capture sequence relationships  
- Some models show validation instability (precision/recall noise)  

---

## 🚀 Applications
- Content moderation (social media, forums)
- Hate speech detection
- Sentiment analysis
- Chatbots and conversational AI
- Recommendation systems

---

## 🛠 Tech Stack
- Python  
- PyTorch  
- TorchText  
- SpaCy  
- Pandas  
- TorchMetrics  
- Weights & Biases  

---

## 📁 Project Structure

toxicity-detection-nlp/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── BERT.ipynb
│   ├── CLSTM.ipynb
│   ├── CNN.ipynb
│   ├── CNN-BiGRU.ipynb
│   ├── MCBiGRU.ipynb
│   └── MLP.ipynb
│
├── src/
│   ├── models/                # Model implementations
│   │   ├── base_model.py
│   │   ├── bert.py
│   │   ├── clstm.py
│   │   ├── cnn.py
│   │   ├── cnn_bigru.py
│   │   ├── mc_bigru.py
│   │   └── mlp.py
│   │
│   └── preprocessing/         # Data preprocessing utilities
│       ├── __init__.py
│       └── utils.py
│
├── report/                    # Project report (LaTeX + PDF)
│   ├── acl2015.pdf
│   ├── acl2015.tex
│   ├── refs.bib
│   └── acl2015.sty
│
├── results/
│   └── metrics/               # Evaluation results (CSV logs)
│
├── data/                      # Dataset (not included)
│   └── README.md
│
└── checkpoints/               # Trained model weights (not included)

---

## 🔮 Future Work
- Integrate transformer-based models beyond BERT  
- Use contextual embeddings instead of static GloVe  
- Improve model interpretability  
- Handle emerging types of toxic language  
- Optimize for real-time inference  

---

## 👤 Author
Seyedamin Hosseini  
Sapienza University of Rome  
