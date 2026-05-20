# Machine Learning in Cybersecurity - Final Project
**Student:** Nishat MM
**Course:** CSC 428 - Machine Learning in Cybersecurity

## Project Overview
This project explores machine learning techniques for cybersecurity, specifically focusing on network intrusion detection using the CICIoT2023 dataset. The analysis covers preprocessing, classical ML models, Deep Learning (LSTM), explainability (SHAP), and adversarial robustness.

## Submission Contents
- **Reports & Presentation**: 
  - `Final_Project_Report.pdf`: Comprehensive course project report.
  - `Nishat_MM_IEEE_Paper.pdf`: Publication-style paper.
  - `Nishat_MM_Final_Project_Slides.pdf`: Presentation slides.
- **Codes/**: Python scripts for the complete pipeline.
- **data/**: Dataset used for training and evaluation.
- **outputs/**: Saved models, evaluation results, and generated figures.

## How to Run
1. **Environment Setup**:
   Ensure you have Python installed. Install the required dependencies using:
   ```bash
   pip install -r Codes/requirements.txt
   ```

2. **Execution Order**:
   The code is structured into 8 sequential steps. Run the scripts in the following order from the root of this folder:
   - `python Codes/1_preprocess.py` (Data cleaning and feature engineering)
   - `python Codes/2_classical_models.py` (Binary classification)
   - `python Codes/3_lstm_model.py` (Deep learning implementation)
   - `python Codes/4_final_summary.py` (Metric comparisons)
   - `python Codes/5_multiclass_models.py` (Multi-class classification)
   - `python Codes/6_shap_analysis.py` (Model interpretability)
   - `python Codes/7_adversarial_robustness.py` (Vulnerability testing)
   - `python Codes/8_publication_summary.py` (Final data for the paper)

## Notes
- All results and figures are automatically saved to the `outputs/` directory.
- The `data/` folder contains a sampled version of the CICIoT2023 dataset (`xxsmall`) for efficiency.
