# Case 02 – Discriminant Analysis: Credit Default Prediction (LendSmart)

## Overview
This case study applies Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) to classify loan applicants as potential defaulters or non-defaulters for LendSmart. The objective is to evaluate both models, compare their performance, and recommend the most appropriate one for business deployment.

This folder contains:
- The Jupyter Notebook with the full technical workflow.
- The corrected Executive Summary.
- The slide deck and the presentation video link.
- All required deliverables for Case 02 of the MA2003B course.

---

## Business Question
How can LendSmart accurately identify applicants with a high probability of default, while minimizing the cost of misclassifications—particularly False Negatives, which correspond to approving clients who will default?

---

## Methods Used
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Train-test split with stratification
- Standardization (no data leakage)
- Box’s M Test for covariance assumptions
- ROC Curve and AUC evaluation

---

## Key Findings
- Both LDA and QDA achieved perfect separation (AUC = 1.0), indicating strong signal in the dataset.
- LDA is preferred due to its interpretability and simplicity.
- The most important predictors were:
  - `payment_history_score` — higher score reduces default risk.
  - `credit_utilization` — higher utilization increases default risk.
  - `job_stability_score` — higher stability reduces default risk.
- False Negative and False Positive definitions were corrected to match credit risk logic.

---

## Deliverables

### 1. Jupyter Notebook  
File: `notebook.ipynb`  
Includes:
- EDA and preprocessing  
- Assumption testing  
- LDA and QDA model training  
- ROC comparison in a single figure  
- Interpretation of coefficients  

### 2. Executive Summary  
File: `Discriminant_analysis (1).pdf `  
A corrected executive report that:
- Defines FP and FN accurately  
- Summarizes model performance  
- Identifies key drivers of default risk  
- Provides recommendations for LendSmart  

### 3. Slide Deck and Presentation  
Slides: `Analytic Dashboard (1).pdf `  
Video: `https://youtu.be/al1uXolgpLY`  
(Video URL must be pasted by you inside this file.)

---

## Files in This Folder
- `notebook.ipynb`
- `Discriminant_analysis (1).pdf `
- `Analytic Dashboard (1).pdf `
- `https://youtu.be/al1uXolgpLY`
- `README.md`

---

## How to Reproduce the Analysis
1. Install dependencies (Python 3.10+ recommended).
2. Install required packages:
pip install numpy pandas scikit-learn matplotlib seaborn
3. Open the notebook:
4. Run all cells in order.
5. Review the outputs and ROC curves.

---

## Authors
Team 3 — Group 602  
- Luis Emilio Fernández González  
- César Isao Pastelin Kohagura  
- Eduardo Botello Casey  


