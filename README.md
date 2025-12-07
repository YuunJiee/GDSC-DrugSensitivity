# 🧬 Genomics of Drug Sensitivity in Cancer (GDSC) - Deep Learning Prediction

> **Course:** CE6146 Introduction to Deep Learning  
> **Date:** December 2025  
> **Goal:** Predict drug sensitivity (IC50) using Dual-Branch Deep Learning to solve the "Cold Start" problem in precision medicine.

---

## 🌟 Overview

This project implements a **Dual-Branch Neural Network** to predict cancer drug response ($IC_{50}$) based on genomic profiles and chemical properties. 

Unlike traditional "One Size Fits All" medicine, this model leverages **Precision Medicine** principles to identify which drugs will work for specific patients. A key focus is the **"Cold Start" problem**—predicting responses for **novel drugs** where no prior biological interaction data exists.

### Key Achievements
*   **Dual-Branch Architecture**: Separate encoders for Cell Line features (Genomics) and Drug features (Chemistry), fused for interaction prediction.
*   **Superior Generalization**: Achieved **$R^2 = 0.65$** on unseen drugs (No IDs), outperforming XGBoost ($R^2 = 0.27$) by **~2.4x**.
*   **Biological Validation**: Correctly identified the **Mitosis Pathway** as the global driver of sensitivity, validated via XGBoost Feature Importance and SHAP.
*   **Optuna Optimization**: Automated hyperparameter tuning (50 trials) identified critical architectural constraints (e.g., wider layers for drug encoding).

---

## 📊 Performance Benchmark

We tested two scenarios:
1.  **With IDs (Memorization)**: Easy task. Model can memorize "Drug A works on Cell B".
2.  **No IDs (Generalization - Cold Start)**: Hard task. Model must learn biological rules.

| Model | No IDs ($R^2$) | With IDs ($R^2$) | Note |
| :--- | :--- | :--- | :--- |
| **XGBoost (Baseline)** | 0.27 | **0.83** | Excellent memorizer, but fails to generalize. |
| **Dual-Branch DL (Ours)** | **0.65** | 0.78 | **Maintains high performance using biological rules.** |

---

## �️ Installation

### 1. Prerequisites
*   Linux / macOS / Windows (WSL2 recommended)
*   Python 3.10+
*   NVIDIA GPU (Recommended for training)

### 2. Setup
Clone the repository and install dependencies:

```bash
# Install Python dependencies
pip install -r requirements.txt

# (Optional) Verify GPU setup
./setup_gpu.sh
```

---

## 🚀 Usage

### 1. Data Preparation
Place the raw dataset (`GDSC_DATASET.csv`) in `data/raw/` (or download from Kaggle).
Then run the preprocessing pipeline:

```bash
python src/data/make_dataset.py
```
*This handles tissue-specific imputation and creates `data/processed/Data_imputed.csv`.*

### 2. Run Experiments (Train & Evaluate)
The main script runs the full pipeline:
1.  **Experiment A (With IDs)**: train/eval baselines & DL.
2.  **Experiment B (No IDs)**: train/eval baselines & DL.
3.  **Visualization**: Generate comparison plots.

```bash
python main.py
```

### 3. Explainability Analysis
To reproduce the SHAP and Importance plots:

```bash
python src/experiments/run_explainability.py
# (Or ensures it runs as part of the main pipeline)
```

---

## � Project Structure

```text
GDSC-DrugSensitivity/
├── data/                   # Raw and Processed Data (Ignored by Git)
├── results/                # Figures, Tables, and Optimization logs
│   ├── figures/            # Learning curves, R2 comparisons, SHAP plots
│   └── optimization/       # Optuna history
├── src/
│   ├── data/               # Preprocessing & Imputation logic
│   ├── models/             # Dual-Branch DL & XGBoost/RF Baselines
│   ├── optimization/       # Optuna Hyperparameter Tuning
│   └── utils/              # Evaluation metrics & Visualization tools
├── main.py                 # Main pipeline entry point
├── setup_gpu.sh            # GPU configuration utility
└── requirements.txt        # Dependency list
```

---

## � Methodology Details

### 1. Feature Engineering (The "Inputs")
*   **Genomic Features**: Gene Expression, Mutation status, Copy Number Variations (CNV).
*   **Drug Features**: Target Pathway, Molecular targets.
*   **Disease Context**: Tissue type (Lung, Skin, Blood, etc.).

### 2. The Model (Dual-Branch)
*   **Cell Encoder**: Deep, narrow layers (handling sparse genomic data).
*   **Drug Encoder**: Wide layers (handling complex chemical interactions).
*   **Fusion**: Concatenation -> Dense -> Output ($LN\_IC_{50}$).

### 3. Optimization
*   **Loss Function**: MSE (Mean Squared Error).
*   **Optimizer**: AdamW with Weight Decay.
*   **Hyperparameter Search**: Optuna (Bayesian Optimization) for Learning Rate, Dropout, and Layer dimensions.

---

## � Contributors

*   **YuunJiee** (Lead Developer & Deep Learning Implementation)
*   *[Other Team Members]*

---

## 📜 License
Educational Project for CE6146.
