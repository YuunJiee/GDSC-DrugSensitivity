# 🧬 CE6146 Final Project — Genomics of Drug Sensitivity in Cancer (GDSC)

> **Presentation Date:** 2025/12/11
> **Goal:** Predict cancer drug sensitivity (IC50) using gene expression and deep learning.
> **Dataset:** [Kaggle - Genomics of Drug Sensitivity in Cancer](https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc)

---

## 🌟 Project Overview

This project aims to build regression models that predict **drug sensitivity (IC50)** values across cancer cell lines using **gene expression and genomic data**.
We will compare traditional ML baselines with deep learning approaches and analyze model interpretability.

---

## 👥 Team Roles

| Member | Role                     | Responsibilities                                        |
| :----- | :----------------------- | :------------------------------------------------------ |
| **A**  | EDA & Data Preprocessing | Data cleaning, merging, normalization, visualization    |
| **B**  | Baseline Models          | Build and evaluate Linear, RF, Lasso, Ridge models      |
| **C**  | Deep Learning Models     | Build MLP / Autoencoder models, tuning, evaluation      |
| **D**  | Result Analysis          | Visualization, report writing, presentation preparation |

---

## 🧩 Workflow

1. Data Preparation (merge & clean GDSC datasets)
2. Baseline Model (Linear, RF, Ridge, Lasso)
3. Deep Learning Model (MLP, Autoencoder)
4. Evaluation (MSE, MAE, R², plots)
5. Reporting & Presentation (visualizations, README, slides)

---

## 🕠 Progress Tracker

| Week  | Dates       | Task                    | Responsible | Status        |
| :---- | :---------- | :---------------------- | :---------- | :------------ |
| **1** | 11/11–11/17 | Data cleaning & EDA     | A           | ⏳ In progress |
| **2** | 11/18–11/24 | Baseline model training | B           | ⏳ Planned     |
| **3** | 11/25–12/01 | Deep learning training  | C           | ⏳ Planned     |
| **4** | 12/02–12/04 | Visualization & summary | D           | ⏳ Planned     |
| **5** | 12/05–12/11 | Final presentation prep | All         | ⏳ Planned     |

> ✅ Updated on: *2025-11-10*

---

## 🗁 Folder Structure

```
GDSC-DrugSensitivity/
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned / merged data
│   └── sample/             # Small demo data
├── notebooks/              # Jupyter notebooks
├── src/                    # Python scripts (preprocess, model, utils)
├── results/                # Metrics and plots
├── reports/                # Summary and findings
├── slides/                 # Final presentation
├── README.md
└── requirements.txt
```

---

## 🔧 Environment

Python 3.10+
Required packages:

```txt
pandas  
numpy  
scikit-learn  
tensorflow  
matplotlib  
seaborn  
openpyxl
```

---

## 📈 Current Status

* [x] Repository initialized
* [ ] Team roles confirmed
* [ ] Data preprocessing started
* [ ] Baseline model training
* [ ] Deep learning model training
* [ ] Evaluation and report

---

## 🧠 Next Steps

* [ ] A: Finish merging datasets and cleaning
* [ ] B: Prepare baseline models notebook
* [ ] C: Set up deep learning notebook structure
* [ ] D: Create PPT outline and example figures

---

## 📜 License

For educational use under CE6146 (Introduction to Deep Learning, NCU CSIE).
