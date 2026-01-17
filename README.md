📌 Multimodal Diagnosis with Medical Images and Clinical Text
Overview

This project implements a multimodal medical diagnosis system that combines chest X-ray images and clinical text to perform binary disease classification. The goal is to evaluate whether fusing heterogeneous medical modalities improves diagnostic performance compared to unimodal models, while maintaining interpretability through explainable AI techniques.

The system is designed with reproducibility, transparency, and explainability in mind, and is suitable for academic evaluation rather than clinical deployment.

Datasets

1. Chest X-ray Images

Dataset: NIH ChestX-ray14 (sampled subset)

Task: Binary classification

Positive: Effusion

Negative: No Finding

Rationale:
Effusion provides sufficient class imbalance and visual complexity for meaningful evaluation while remaining computationally feasible in Google Colab.

2. Clinical Text

Dataset: PubMed-QA (labeled subset)

Task: Binary classification (Yes / No)

Rationale:
Used as a proxy for clinical notes due to access restrictions on full MIMIC-III data. This avoids paid training requirements while preserving academic validity.

⚠️ Note:
Image and text samples are not paired at the patient level. This project evaluates architectural multimodal fusion, not patient-level clinical inference.

Project Structure
multimodal-dx/
│
├── src/
│ ├── datasets/ # Image, text, and multimodal datasets
│ ├── models/ # Image encoder, text encoder, fusion models
│ ├── training/ # Training & evaluation scripts
│ ├── explainability/ # Grad-CAM implementation
│
├── reports/
│ ├── figures/ # Grad-CAM visualisations
│ ├── tables/ # Metrics and curated indexes
│
├── README.md
└── requirements.txt

Models
Image Model

Backbone: DenseNet-121 (ImageNet pretrained)

Input: Chest X-ray (224×224)

Output: Binary classification logits

Text Model

Encoder: Clinical BERT-style transformer

Input: Tokenised clinical text

Output: Binary classification logits

Fusion Models

Two fusion strategies are implemented:

Concatenation Fusion

Attention-based Fusion

Fusion operates at the feature level, combining latent representations from image and text encoders.

Explainability
Grad-CAM

Applied to the image encoder

Highlights spatial regions influencing predictions

Implemented with safe backward hooks (no in-place ops)

Curated Explainability Set

Grad-CAM visualisations are generated for:

True Positives (TP)

True Negatives (TN)

False Positives (FP)

False Negatives (FN)

A CSV index maps:

Sample index

True label

Predicted label

Prediction probability

Output figure path

📁 Outputs:

reports/figures/gradcam_curated/
reports/tables/gradcam_curated_index.csv

Training & Evaluation
Image Model
python -m src.training.train_image

Text Model
python -m src.training.train_text

Multimodal Fusion
python -m src.training.train_fusion

Fusion training includes:

ROC-AUC

PR-AUC (important for class imbalance)

F1-score with validation-tuned threshold

Confusion matrices

Explainability Execution
Random Samples
python -m src.training.run_gradcam

Curated TP / TN / FP / FN
python -m src.training.run_gradcam_curated

Reproducibility

Fixed random seeds (PyTorch, NumPy, Python)

Deterministic dataset splits

Metrics saved to disk

Models checkpointed by best validation score

Ethical & Practical Considerations

Uses public and demo datasets only

No protected patient data

Not intended for clinical use

Results reflect research exploration, not medical advice

Limitations

Image and text data are not patient-aligned

Text dataset is a proxy for real clinical notes

Dataset sizes are intentionally constrained

Performance prioritises analysis over optimisation

Dependencies

Install required packages:

pip install -r requirements.txt

Author

Qasim Bilal Arshad
Final-Year Project — Multimodal Machine Learning
