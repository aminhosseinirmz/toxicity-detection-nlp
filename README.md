# Toxicity Detection NLP

This project compares several deep learning architectures for multi-label toxic comment classification on the Wikipedia / Jigsaw toxic comment dataset.

## Models
- CNN
- C-LSTM
- CNN-BiGRU
- MCBiGRU
- MLP
- BERT

## Project structure
- `notebooks/` experiment notebooks
- `src/models/` model implementations
- `src/preprocessing/` preprocessing utilities
- `report/` LaTeX report and PDF
- `results/metrics/` saved metric logs

## Dataset
The dataset is not included in this repository.

Download:
- `train.csv`
- `test.csv`
- `test_labels.csv`

from the Kaggle Jigsaw Toxic Comment Classification Challenge and place them in the `data/` folder.

## Main libraries
- PyTorch
- TorchText
- TorchMetrics
- Transformers
- pandas
- spaCy
- scikit-learn
- Weights & Biases
