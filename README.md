# Metabolic Syndrome Risk Identification with Environmental Exposure Data (KNHANES 2018–2022)

Files and guides to reproduce the experiments of the paper:

> **Development of Metabolic Syndrome Risk Identification Model by Integrating Korea National Health and Nutrition Examination Survey Data with Regional Particulate Matter, Air Pollutants, and Green Environmental Factors**, BMC Public Health

By combining regional air pollution (CO, NO2, O3, PM10, PM2.5, SO2) and green area per capita (green_1) data with the National Health and Nutrition Examination Survey (KNHANES) 2018–2022 data (over 30 years old, N = 20,035), this study predicts the risk of metabolic syndrome (MetS) with eight machine learning/deep learning models (Logical Regression, Random Forest, XGBoost, LightGBM, SVM, Basic MLP, DeepMLP, and ResNetMLP).

## 📝 Authors
Mukeun Choi_1 and Taeyeon Oh_2*

* 1 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* 1 : SDG Management School, Geneva, Switzerland
* 2 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* \* : Corresponding Author

---

## 📦 File Path

```
.
├── data/                          # Preprocessed Dataset for Analysis (KNHANES raw data is not included)
├── models/                        # Trained models
├── results/                       # Experiment results (csv)
│   └── figures/                   # Visualized results (png)
└── scripts/                       # Experiment and visualize code scripts
```

### 🔍 Experiment Code in scripts path

| File | Description |
|------|------|
| 01_Data_Preprocess.py | Data Preprocessing |
| 02_Model_Training.py | Model Training |
| 03_Experiments_01.py | Experiments #01 |
| 04_Experiments_02.py | Experiments #02 |
| 05_Visualization_01.py | Visualization #01 |
| 06_Visualization_02.py | Visualization #02 |

### 📚 Reference File

| File | Description |
|------|------|
| `README.md` | This File |

---

## 🔥 Key Results

### 1) 8 Model Overall Performance Rankings

| Rank | Model | AUC | AUPRC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 1 | DeepMLP | 0.6664 | 0.3973 | 0.4797 | 0.5866 | 0.3686 | 0.6868 |
| 2 | Random Forest | 0.6623 | 0.3935 | 0.4556 | 0.6056 | 0.3692 | 0.5947 |
| 3 | XGBoost | 0.6591 | 0.3920 | 0.4730 | 0.5874 | 0.3663 | 0.6673 |
| 4 | LightGBM | 0.6582 | 0.3892 | 0.4733 | 0.5807 | 0.3633 | 0.6789 |
| 5 | BasicMLP | 0.6562 | 0.3923 | 0.4786 | 0.5814 | 0.3657 | 0.6924 |
| 6 | Logistic Regression | 0.6529 | 0.3674 | 0.4675 | 0.6177 | 0.3810 | 0.6049 |
| 7 | ResNetMLP | 0.6465 | 0.3819 | 0.4667 | 0.5764 | 0.3586 | 0.6679 |
| 8 | SVM | 0.6438 | 0.3821 | 0.0252 | 0.7242 | 0.6573 | 0.0128 |

### 2) Statistical Significance Between Models (DeLong test)
- The difference in AUC between the top three models (DeepMLP, Random Forest, and XGBoost) **All are not statistically significant**(p>0.5) → the basis for reporting the top three side-by-side instead of "one representative model"
- Significant (p<0.05) difference among all 28 pairs is 8 pairs, mostly in comparison between lower-ranked and higher-ranked models, such as SVM, Logistic Regression, and ResNetMLP

### 3) Calibration
| Model | Brier Score | Calib. Slope | Calib. Intercept |
|---|---|---|---|
| DeepMLP | 0.2270 | 0.6379 | −0.0132 |
| Random Forest | 0.2235 | 0.9220 | −0.1464 |
| XGBoost | 0.2287 | 0.7161 | −0.0565 |

Random forest is the calibration slope (0.922) closest to the ideal 1, resulting in the best correction of the predictive probability.

### 4) HE_obe(Obesity) information leakage when variables are included

| Condition | DeepMLP | Random Forest | XGBoost |
|---|---|---|---|
| HE_obe excluded (final model) | 0.6548 | 0.6623 | 0.6591 |
| HE_obe included | 0.8184 | 0.8117 | 0.8210 |

With HE_obe included, the AUC surges by approximately 0.15 to 0.16 points because HE_obe itself effectively overlaps (predictor-result variable leakage) with the metabolic syndrome diagnostic criteria, thus excluding HE_obe in the final model.

### 5) PCA/SPCA vs Raw environment variables
| Model | Feature Set | AUC |
|---|---|---|
| Random Forest | 7 raw variables | 0.6604 |
| Random Forest | 3 PCA components | 0.6598 |
| XGBoost | 7 raw variables | 0.6580 |
| XGBoost | 3 PCA components | 0.6617 |
| DeepMLP | 7 raw variables | 0.6628 |
| DeepMLP | 3 PCA components | 0.6639 |

The difference in AUC between the two methods is only 0.0006 to 0.0037, so we did not adopt PCA/SPCA in the final model on the basis that the interpretability of the original variable can be maintained without loss of performance.

### 6) Relative Contribution of green_1 (Green per capita) among Environmental Variables

| Model | Top Variables | green_1 SHAP ranking | Rates to Top Variables |
|---|---|---|---|
| DeepMLP | sex | 8th | 8.69% |
| Random Forest | sex | 16th | 4.45% |
| XGBoost | sex | 13th | 6.44% |

The contribution of green_1 is 4-9% compared to the highest individual factors such as gender, which should be carefully interpreted as a **association factor** considered along with other higher variables rather than concluding it as a "significant mitigating factor" (based on the avoidance of causal expression).

### 7) Other Sensitivity Analysis Summary
- **Whether the survey weight is reflected**: The AUC difference between weighted/unweighted conditions is very small, up to 0.005 → Predictive performance is not very sensitive to weight reflection
- **Leave-One-Region-Out (LORO) geographical verification**: The mean AUC difference from random segmentation is up to 0.015 (statistically difficult to distinguish when considering the standard deviation 0.025 to 0.028) → Support that regional level information leakage concerns are not serious
- **5-Fold CV vs Single fold**: The AUC difference between the two methods is slight and not consistent
- **Multicollinearity**: High correlation (|r|≥0.7) in the combination of PM10–PM2.5, NO2–O3–green_1 recommends cluster-wise interpretation rather than individual variable SHAP/regression coefficient rankings


---

## 🚀 Quick Start
1) Login to Google Colab environment (NVIDIA T4 GPU 25GB RAM)
2) Copy and paste Python files in the "scripts" folder into each cell in order of number
3) Upload 2 preprocessed data files to your Google Drive folder
4) Connect Google Drive to your colab file
5) Set up and verify the path in the code
6) Run each cell's code in order
7) You can check and download the results

## 🔬 Required Packages

```bash
pip install numpy pandas scikit-learn torch xgboost lightgbm shap optuna imbalanced-learn scipy matplotlib joblib
```


## ⚠️ Reproducibility Notes

- **Random Seed**: `RANDOM_STATE = 42` (The deep learning model also sets `torch.manual_seed` and `cudn.detergentic=True` together)
- **Correction of class imbalance**: SMOTE applied to training data, deep learning model uses `IMBALANCE_RATIO = 2.54` as `pos_weight`
- **Hyperparameter Search**: Optuna(TPE sampler) + Stratified 5-Fold CV, See `results/0_optuna_search_space.csv` for search range
- **Evaluation indicators**: Use AUC-ROC as the basic optimization/comparison indicator and view AUPRC, F1, Precision, Recall, Brier score, calibration scope/intercept together
- **Test for statistical significance**: Use DeLong test(28 pairs of 8 models) and bootstrap test (top 3 model pair) together


---


## 📝 Cite this Article

Choi, M., Oh, T. Development of metabolic syndrome identification model by integrating Korea national health and nutrition examination survey data with regional particulate matter, air pollutants, and green environmental factors. BMC Public Health (2026). https://doi.org/10.1186/s12889-026-29087-1

