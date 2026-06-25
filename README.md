# IoT Network Intrusion Detection

**ML pipeline for network intrusion detection on IoT traffic — 5 classifiers, 508K instances, 34 attack types, SHAP explainability, and adversarial robustness testing.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dataset: CICIoT2023](https://img.shields.io/badge/dataset-CICIoT2023-orange)](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---

## Overview

IoT devices are increasingly targeted by network attacks, yet most intrusion detection research uses clean, balanced datasets that don't reflect production conditions. This project builds a full ML pipeline on CICIoT2023 — one of the most comprehensive IoT attack datasets available — covering binary and multi-class classification across 34 attack types and 508K network traffic instances.

The pipeline goes beyond accuracy: it includes SHAP-based feature attribution to explain model decisions and adversarial robustness testing to surface model vulnerabilities under crafted inputs.

---

## Dataset

**CICIoT2023** — Canadian Institute for Cybersecurity
- 508,000 network traffic instances
- 34 attack categories (DDoS, DoS, Reconnaissance, Mirai, Brute Force, and more)
- 46 numerical features extracted from raw packet captures
- Heavily imbalanced: attack traffic ~83%, benign ~17%

---

## Pipeline

```
Raw CICIoT2023 Data
        │
        ▼
1. Preprocessing     — null removal, label encoding, SMOTE oversampling
        │
        ▼
2. Binary Classification   — benign vs. attack (5 models)
        │
        ▼
3. Multi-Class Classification  — 34 attack type labels (5 models)
        │
        ▼
4. SHAP Analysis      — feature attribution for XGBoost predictions
        │
        ▼
5. Adversarial Testing  — FGSM-style perturbations on neural network
        │
        ▼
6. Summary Report     — cross-model metric comparison
```

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear baseline | L2 regularization |
| Random Forest | Ensemble | 100 estimators |
| XGBoost | Gradient boosting | Primary model for SHAP analysis |
| LightGBM | Gradient boosting | Fast training on full dataset |
| LSTM | Deep learning | Sequential packet-level features |

---

## Results (Binary Classification)

| Model | Accuracy | F1 (Attack) | AUC-ROC |
|-------|----------|-------------|---------|
| Logistic Regression | 91.2% | 0.943 | 0.961 |
| Random Forest | 99.1% | 0.993 | 0.999 |
| XGBoost | 99.3% | 0.994 | 0.999 |
| LightGBM | 99.2% | 0.993 | 0.999 |
| LSTM | 98.7% | 0.989 | 0.997 |

**Key finding:** Tree-based ensembles (XGBoost, LightGBM, RF) achieve near-perfect detection with sub-second inference — making them practical for edge gateway deployment.

---

## Explainability (SHAP)

SHAP values reveal which network features drive attack classification:

- **Top features:** `flow_duration`, `tot_fwd_pkts`, `fwd_pkt_len_mean`, `bwd_pkt_len_max`
- DDoS attacks are primarily distinguished by packet rate and flow duration
- Reconnaissance attacks show distinct patterns in forward/backward byte ratios

Full SHAP summary plots and force plots are saved to `outputs/shap/`.

---

## Adversarial Robustness

Applied FGSM-style feature perturbations (epsilon = 0.01–0.1) to the LSTM model:

- At epsilon = 0.05, attack detection F1 drops from 0.989 to 0.871
- Suggests LSTM is more brittle than tree-based models under input perturbation
- Tree models (XGBoost) show significantly higher adversarial stability

---

## Quick Start

```bash
git clone https://github.com/nishat123456/iot-intrusion-detection
cd iot-intrusion-detection
pip install -r Codes/requirements.txt
```

Run the pipeline in order:

```bash
python Codes/1_preprocess.py          # Data cleaning + feature engineering
python Codes/2_classical_models.py    # Binary classification
python Codes/3_lstm_model.py          # Deep learning
python Codes/4_final_summary.py       # Metric comparison
python Codes/5_multiclass_models.py   # 34-class classification
python Codes/6_shap_analysis.py       # Feature attribution
python Codes/7_adversarial_robustness.py  # Robustness testing
python Codes/8_publication_summary.py # Final report data
```

All outputs (models, figures, metrics) are saved to `outputs/`.

---

## References

- [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — University of New Brunswick
- Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
- Goodfellow et al. (2014) — Explaining and Harnessing Adversarial Examples

---

**Author:** Mustaqim Nishat | [nishat12sikdar@gmail.com](mailto:nishat12sikdar@gmail.com)
