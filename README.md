# Multimodal Deep Learning for Pneumonia Detection

A lightweight, reproducible and explainable multimodal deep learning system for pneumonia detection from paired chest X-ray images and radiology reports.

## Overview

This project was developed as an MSc Artificial Intelligence dissertation. It investigates whether combining visual information from chest X-rays with clinical information contained in radiology reports can provide a more robust diagnostic classification system than image-only approaches.

The system uses:

- **DenseNet-121** for chest X-ray feature extraction
- **BioClinicalBERT** for clinical text representation
- **Concatenation Fusion** as a simple multimodal baseline
- **Gated Fusion** to learn the relative contribution of image and text modalities
- **Grad-CAM** for visual explanations
- **Token occlusion** for text-level explanations
- **Patient-level splitting and fixed random seeds** for leakage prevention and reproducibility
- **CPU-only execution** to keep the system accessible without specialised hardware

> **Research-only notice:** This repository implements an academic research system and is not a clinical diagnostic device. Predictions should not be used for patient diagnosis or treatment decisions.

## Research Motivation

Chest X-rays provide important spatial information but can be ambiguous because of overlapping anatomy and subtle abnormalities. Radiology reports provide complementary semantic and clinical context.

A major methodological risk in multimodal medical AI is **label leakage**: if diagnostic conclusions are included directly in model input, a model can learn the answer rather than the underlying task.

This project therefore separates report sections:

- **Impression** → used to derive diagnostic labels
- **Findings** → used as model input

Patient-level dataset splitting is also used to reduce leakage between training and evaluation data.

## System Architecture

```text
                    IU X-Ray Dataset
                           |
             +-------------+-------------+
             |                           |
        Chest X-Ray                 Radiology Report
             |                           |
             v                           v
       DenseNet-121               BioClinicalBERT
             |                           |
             v                           v
       Image Features              Text Features
             |                           |
             +-------------+-------------+
                           |
                    Fusion Engine
                    /           \
                   /             \
        Concatenation         Gated Fusion
                   \             /
                    \           /
                     v         v
                    Prediction
                        |
                        v
              Pneumonia / Normal
                        |
              +---------+---------+
              |                   |
           Grad-CAM          Token Occlusion
              |                   |
              v                   v
       Image attribution     Text attribution
```

## Models

### Image Encoder

A pretrained **DenseNet-121** model is used to extract visual representations from chest X-ray images.

DenseNet-121 was selected as a parameter-efficient CNN backbone with established relevance to chest X-ray classification.

### Text Encoder

**BioClinicalBERT** is used to encode radiology report text and capture clinical terminology, context and linguistic patterns.

### Fusion Strategies

Two multimodal strategies are evaluated:

#### 1. Concatenation Fusion

Image and text feature vectors are directly combined before classification.

#### 2. Gated Fusion

A learnable gate dynamically weights the contribution of the image and text representations before classification.

The purpose is to allow the model to adjust its reliance on each modality rather than assuming that both sources are equally informative for every case.

## Dataset and Data Handling

The project uses the publicly available **IU X-Ray dataset**, containing paired chest X-ray images and radiology reports.

The preprocessing pipeline was designed around three principles:

1. **Leakage safety** — diagnostic labels are derived from report impressions while model input is restricted to findings.
2. **Reproducibility** — fixed seeds and deterministic data splitting are used.
3. **Controlled preprocessing** — input canonicalisation, validation and file integrity checks are applied before training.

The classification task is:

```text
Pneumonia vs Normal
```

## Evaluation

The system is evaluated using metrics designed to provide a more informative view of imbalanced medical classification:

- ROC-AUC
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Sensitivity
- Specificity

Models are evaluated across multiple random seeds to reduce dependence on a single favourable initialisation.

### Reported Results

| Model                      |  Test AUC | Balanced Accuracy |
| -------------------------- | --------: | ----------------: |
| Image-only                 |    0.7127 |            0.6334 |
| Text-only                  |    0.8579 |            0.7606 |
| Early/Concatenation Fusion |     0.848 |             0.739 |
| Gated Fusion               | **0.849** |         **0.760** |

The gated fusion experiment produced a sensitivity of **0.727** in the reported single-seed evaluation, compared with **0.636** for early fusion.

The multi-seed analysis further examined variation caused by random initialisation. The dissertation reports that gated fusion showed lower variance in MCC and more consistent sensitivity than early concatenation fusion.

## Explainability

Explainability is treated as a core part of the system rather than an additional visualisation step.

### Grad-CAM

Grad-CAM is used to generate heatmaps showing regions of the chest X-ray that contributed to the visual model's prediction.

### Token Occlusion

Individual tokens in the radiology report can be masked to measure their effect on the prediction score. This provides an indication of which textual information influenced the model.

### Gate Analysis

The gated fusion mechanism records its learned weighting behaviour, allowing analysis of whether individual predictions rely more heavily on the image or text modality.

## Reproducibility and Engineering

The implementation was designed as a software engineering artefact rather than a collection of experimental scripts.

Key engineering principles include:

- Modular separation of data, encoders, fusion and prediction components
- Fixed random seeds
- Patient-level data splitting
- Persistent experiment logging
- Unit testing
- Input validation
- CPU-compatible training and inference
- Decoupled fusion modules
- Explicit handling of data quality and file integrity

## Technology Stack

| Area            | Technology                                 |
| --------------- | ------------------------------------------ |
| Language        | Python 3.8                                 |
| Deep Learning   | PyTorch                                    |
| Computer Vision | Torchvision, DenseNet-121                  |
| Clinical NLP    | Hugging Face Transformers, BioClinicalBERT |
| Evaluation      | Scikit-learn                               |
| Data Processing | NumPy, Pandas                              |
| Explainability  | Grad-CAM, token occlusion                  |
| Testing         | Pytest                                     |

## Key Findings

The project found that:

- Multimodal fusion provided a substantial improvement over the image-only baseline.
- Gated fusion achieved similar overall discrimination to early fusion while producing a different sensitivity/specificity trade-off.
- Multi-seed evaluation was important for understanding robustness rather than relying on a single training run.
- Clinical text was highly informative, highlighting both the value of multimodal context and the importance of preventing label leakage.
- Explainability exposed clinically relevant failure modes involving label ambiguity and overlap between pneumonia and chronic lung disease.
- Meaningful multimodal experimentation was possible using standard CPU hardware and frozen pretrained encoders.

## Limitations

The system is a research prototype and has important limitations.

- The IU X-Ray dataset does not represent the full diversity of clinical populations and imaging environments.
- Radiology reports contain ambiguity and annotation noise.
- CPU-only execution limited the amount of hyperparameter tuning and experimentation that could be performed.
- Explainability methods such as Grad-CAM and token occlusion provide evidence about model behaviour but do not prove causal reasoning.
- The system has not undergone prospective clinical validation.
- The model should not be interpreted as a clinically deployable diagnostic system.

## Future Work

Potential extensions include:

- Evaluation on larger and more diverse datasets
- External validation across institutions
- More extensive hyperparameter optimisation
- Investigation of additional multimodal fusion mechanisms
- Improved handling of missing or noisy modalities
- Calibration and uncertainty estimation
- More systematic clinical validation of explanations
- GPU-enabled experimentation for broader model comparisons

## Project Context

**MSc Artificial Intelligence — Kingston University**

The project was completed as an MSc dissertation and combines:

- Deep learning
- Computer vision
- Clinical NLP
- Multimodal learning
- Experimental design
- Explainable AI
- Data leakage prevention
- Software engineering
- Robust evaluation

## Author

**Qasim Bilal Arshad**

MSc Artificial Intelligence, Kingston University  
BSc Computer Science, University of Leicester

LinkedIn: www.linkedin.com/in/qasim-bilal-arshad
