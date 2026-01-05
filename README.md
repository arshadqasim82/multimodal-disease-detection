# Multimodal Deep Learning for Disease Diagnosis

This project implements a modular multimodal diagnostic pipeline combining:

- CNN-based feature extraction for medical images
- ClinicalBERT-based feature extraction for clinical text
- Fusion strategies (concatenation and attention)
- Explainability (Grad-CAM + attention visualisation)
- Fairness and subgroup evaluation

## Structure

See `/src` for datasets, models, training, evaluation, and explainability modules.

## Running (Colab recommended)

Training scripts are in `/src/training`.
Configuration is in `/configs`.
