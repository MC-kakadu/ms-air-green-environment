# Install library on Google Colab, if the experiment session was restarted. 
# !pip install -q shap optuna imbalanced-learn xgboost lightgbm --break-system-packages

# Google Drive import
from google.colab import drive
drive.mount('/content/drive')


import os, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, accuracy_score,
                              confusion_matrix, brier_score_loss, precision_recall_curve)
from sklearn.calibration import calibration_curve
import scipy.stats as stats

BASE       = '/content/drive/MyDrive/MetS_Experiment/'
MODEL_DIR  = f'{BASE}/models'
DATA_DIR   = f'{BASE}/data/'
RESULT_DIR = f'{BASE}/results'          
FULL_DATA_PATH        = f'{BASE}/data/knhanes_encoded_final.csv'
FULL_DATA_PATH_W_OBE  = f'{BASE}/data/knhanes_encoded_final_add_HEobe.csv'
os.makedirs(RESULT_DIR, exist_ok=True)

TOP_K = 3                       
RANDOM_STATE = 42
IMBALANCE_RATIO = 2.54          

meta_cols = ['year', 'wt_itvex', 'wt_pool', 'kstrata', 'psu']
continuous_cols = ['age', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 'green_1']

def get_X_y_w(df, has_weight=True):
    y = df['MS'].values
    w = df['wt_pool'].values if has_weight else np.ones(len(df))
    cols_to_drop = ['MS'] + [c for c in meta_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X, y, w

train_orig  = pd.read_csv(f'{DATA_DIR}/train_original.csv')
train_smote = pd.read_csv(f'{DATA_DIR}/train_smote.csv')
val_df      = pd.read_csv(f'{DATA_DIR}/validation_set.csv')
test_df     = pd.read_csv(f'{DATA_DIR}/test_set.csv')

X_train, y_train, w_train             = get_X_y_w(train_orig, has_weight=True)
X_train_smote, y_train_smote, w_train_smote = get_X_y_w(train_smote, has_weight=False)
X_val,   y_val,   w_val               = get_X_y_w(val_df, has_weight=True)
X_test,  y_test,  w_test              = get_X_y_w(test_df, has_weight=True)

scaler = joblib.load(f'{MODEL_DIR}/standard_scaler.pkl')
X_train_scaled       = X_train.copy();       X_train_scaled[continuous_cols]       = scaler.transform(X_train[continuous_cols])
X_train_smote_scaled = X_train_smote.copy(); X_train_smote_scaled[continuous_cols] = scaler.transform(X_train_smote[continuous_cols])
X_val_scaled         = X_val.copy();         X_val_scaled[continuous_cols]         = scaler.transform(X_val[continuous_cols])
X_test_scaled        = X_test.copy();        X_test_scaled[continuous_cols]        = scaler.transform(X_test[continuous_cols])
in_dim = X_test_scaled.shape[1]

print(f"Data load completed: Train {len(X_train)} / Val {len(X_val)} / Test {len(X_test)}, the number of input features = {in_dim}")

# The structure of Deep Learning models
class BasicMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                  nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)

class DeepMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.layer(x) + x)

class ResNetMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.res_blocks = nn.Sequential(ResBlock(64), ResBlock(64), ResBlock(64))
        self.fc_out = nn.Linear(64, 1)
    def forward(self, x): return self.fc_out(self.res_blocks(self.fc_in(x)))

DL_CLASSES = {'BasicMLP': BasicMLP, 'DeepMLP': DeepMLP, 'ResNetMLP': ResNetMLP}
ML_NAMES   = ['Logistic_Regression', 'Random_Forest', 'XGBoost', 'LightGBM', 'SVM']

def load_dl_model(model_class, path, in_dim):
    m = model_class(in_dim)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

def get_dl_probs(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.FloatTensor(X.values))).numpy().flatten()

def train_dl_model(model_class, X_tr, y_tr, w_tr, X_v, y_v, w_v, pos_weight=None, patience=15, seed=RANDOM_STATE):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr.values), torch.FloatTensor(y_tr).unsqueeze(1),
                                             torch.FloatTensor(w_tr).unsqueeze(1)), batch_size=256, shuffle=True, generator=g)
    val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_v.values), torch.FloatTensor(y_v).unsqueeze(1),
                                             torch.FloatTensor(w_v).unsqueeze(1)), batch_size=256, shuffle=False)
    model = model_class(X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), reduction='none') if pos_weight else nn.BCEWithLogitsLoss(reduction='none')
    best_val_loss, patience_counter, best_weights = float('inf'), 0, None
    for epoch in range(200):
        model.train()
        for b_X, b_y, b_w in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            (loss * b_w).mean().backward()
            optimizer.step()
        scheduler.step()
        model.eval(); val_loss = 0
        with torch.no_grad():
            for b_X, b_y, b_w in val_loader:
                val_loss += (criterion(model(b_X), b_y) * b_w).mean().item()
        if val_loss < best_val_loss:
            best_val_loss, patience_counter, best_weights = val_loss, 0, model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience: break
    if best_weights is not None: model.load_state_dict(best_weights)
    return model

def make_ml_models():
    return {
        'Logistic_Regression': LogisticRegression(C=0.0106, penalty='l2', random_state=RANDOM_STATE, class_weight='balanced'),
        'Random_Forest': RandomForestClassifier(n_estimators=380, max_depth=8, min_samples_split=6, max_features='sqrt', random_state=RANDOM_STATE, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=580, max_depth=4, learning_rate=0.0101, subsample=0.923, colsample_bytree=0.755, random_state=RANDOM_STATE, eval_metric='logloss', scale_pos_weight=IMBALANCE_RATIO),
        'LightGBM': LGBMClassifier(n_estimators=250, num_leaves=20, learning_rate=0.0160, min_child_samples=10, subsample=0.802, colsample_bytree=0.845, random_state=RANDOM_STATE, class_weight='balanced'),
        'SVM': SVC(C=0.105, gamma='scale', probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
    }

# 8-trained models load (no re-training)
predictions, ml_models, dl_models = {}, {}, {}
for name in ML_NAMES:
    ml_models[name] = joblib.load(f'{MODEL_DIR}/{name}.pkl')
    predictions[name] = ml_models[name].predict_proba(X_test_scaled)[:, 1]
for name, cls in DL_CLASSES.items():
    dl_models[name] = load_dl_model(cls, f'{MODEL_DIR}/{name}.pth', in_dim)
    predictions[name] = get_dl_probs(dl_models[name], X_test_scaled)

rank_df = pd.Series({k: roc_auc_score(y_test, v, sample_weight=w_test) for k, v in predictions.items()}) \
            .sort_values(ascending=False).rename('AUC').reset_index().rename(columns={'index': 'Model'})
rank_df.insert(0, 'Rank', range(1, len(rank_df) + 1))
rank_df.to_csv(f'{RESULT_DIR}/0_model_ranking_all8.csv', index=False)
top_models = rank_df['Model'].head(TOP_K).tolist()
print(rank_df.to_string(index=False))
print(f"\n✅ Top-{TOP_K} model: {top_models}")

# Optuna search range
optuna_space = pd.DataFrame([
    ['Logistic_Regression', 'C', '1e-4 ~ 10 (log-uniform)'],
    ['Random_Forest', 'n_estimators / max_depth / min_samples_split', '100-500 / 3-15 / 2-10'],
    ['XGBoost', 'n_estimators / max_depth / learning_rate / subsample / colsample_bytree', '100-800 / 3-10 / 1e-3-0.3 / 0.5-1.0 / 0.5-1.0'],
    ['LightGBM', 'n_estimators / num_leaves / learning_rate / min_child_samples', '100-500 / 10-60 / 1e-3-0.3 / 5-50'],
    ['SVM', 'C / gamma', '1e-3-10 (log) / scale, auto'],
], columns=['Model', 'Hyperparameters', 'Search_Range'])
optuna_space.to_csv(f'{RESULT_DIR}/0_optuna_search_space.csv', index=False)


# [PART 1] Test statistical significance of performance differences between models
# DeLong test + Weighted Paired Bootstrap (wt_pool)
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def delong_roc_variance(y_true, y_prob):
    order = (-y_true).argsort()
    y_prob_sorted = y_prob[order]
    pos_n = int(y_true.sum()); neg_n = len(y_true) - pos_n
    pos_scores, neg_scores = y_prob_sorted[:pos_n], y_prob_sorted[pos_n:]
    tx = compute_midrank(pos_scores); ty = compute_midrank(neg_scores); tz = compute_midrank(y_prob_sorted)
    v01 = (tz[:pos_n] - tx) / neg_n
    v10 = 1 - (tz[pos_n:] - ty) / pos_n
    auc = tz[:pos_n].sum() / (pos_n * neg_n) - (pos_n + 1) / (2 * neg_n)
    return auc, v01, v10

def delong_roc_test(y_true, prob_a, prob_b):
    auc_a, v01_a, v10_a = delong_roc_variance(y_true, prob_a)
    auc_b, v01_b, v10_b = delong_roc_variance(y_true, prob_b)
    var_a = np.var(v01_a) / len(v01_a) + np.var(v10_a) / len(v10_a)
    var_b = np.var(v01_b) / len(v01_b) + np.var(v10_b) / len(v10_b)
    cov = (np.cov(v01_a, v01_b)[0, 1] / len(v01_a)) + (np.cov(v10_a, v10_b)[0, 1] / len(v10_a))
    se = np.sqrt(var_a + var_b - 2 * cov)
    z = (auc_a - auc_b) / se if se > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, z, p

from itertools import combinations
delong_rows = []
for m1, m2 in combinations(rank_df['Model'], 2):
    auc1, auc2, z, p = delong_roc_test(y_test, predictions[m1], predictions[m2])
    delong_rows.append([f'{m1} vs {m2}', round(auc1, 4), round(auc2, 4), round(z, 4), round(p, 4),
                         'Significant (p<0.05)' if p < 0.05 else 'Not Significant'])
delong_df = pd.DataFrame(delong_rows, columns=['Comparison', 'AUC_1', 'AUC_2', 'Z', 'p_value', 'Result'])
delong_df.to_csv(f'{RESULT_DIR}/1_delong_test_all_pairs.csv', index=False)
n_sig = (delong_df['Result'] == 'Significant (p<0.05)').sum()
print(f"[PART 1] DeLong test Completed: significant pair {n_sig} of Total {len(delong_df)} pair ")

def weighted_bootstrap_test(y, w, prob_a, prob_b, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        auc_a = roc_auc_score(y[idx], prob_a[idx], sample_weight=w[idx])
        auc_b = roc_auc_score(y[idx], prob_b[idx], sample_weight=w[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return diffs.mean(), ci_lo, ci_hi, p

boot_rows = []
for m1, m2 in combinations(top_models, 2):   
    mean_diff, lo, hi, p = weighted_bootstrap_test(y_test, w_test, predictions[m1], predictions[m2])
    boot_rows.append([f'{m1} vs {m2}', round(mean_diff, 4), round(lo, 4), round(hi, 4), round(p, 4)])
boot_df = pd.DataFrame(boot_rows, columns=['Comparison', 'Mean_AUC_Diff', 'CI_Lower', 'CI_Upper', 'p_value'])
boot_df.to_csv(f'{RESULT_DIR}/1_bootstrap_test_top3_pairs.csv', index=False)
print(boot_df.to_string(index=False))
print(f"[PART 1] 'DeLong test result - significant pair {n_sig} of 8-models {len(delong_df)} pair, "
      f"The difference in performance between AUC Top-3 model({', '.join(top_models)}) was not statistically significant.'")


# [PART 2] Confusion Matrix + Calibration (Top-3 model)
fig, axes = plt.subplots(1, TOP_K, figsize=(5 * TOP_K, 4.5))
for ax, name in zip(axes, top_models):
    pred = (predictions[name] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
plt.tight_layout()
plt.savefig(f'{RESULT_DIR}/2_confusion_matrix_top3.png', dpi=300)
plt.show()

calib_rows = []
plt.figure(figsize=(7, 7))
for name in top_models:
    prob = predictions[name]
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10, strategy='quantile')
    slope, intercept, *_ = stats.linregress(prob_pred, prob_true)
    brier = brier_score_loss(y_test, prob)
    calib_rows.append([name, round(brier, 4), round(slope, 4), round(intercept, 4)])
    plt.plot(prob_pred, prob_true, marker='o', label=f'{name} (Brier={brier:.4f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
plt.title('Calibration Plot - Top 3 Models'); plt.xlabel('Mean Predicted Probability'); plt.ylabel('Fraction of Positives')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f'{RESULT_DIR}/2_calibration_plot_top3.png', dpi=300)
plt.show()
pd.DataFrame(calib_rows, columns=['Model', 'Brier_Score', 'Calib_Slope', 'Calib_Intercept']) \
    .to_csv(f'{RESULT_DIR}/2_calibration_top3.csv', index=False)
print(f"[PART 2] Confusion Matrix / Calibration(Brier, slope, intercept) - Top-{TOP_K} model saved")


# [PART 3] Subgroup Analysis (by sex / age) - Top-3 model
sex_col = [c for c in X_test.columns if 'sex' in c][0]
sex_data = X_test[sex_col].values
age_data = X_test['age'].values
subgroups = {
    'Total': np.ones(len(y_test), dtype=bool),
    'Male': (sex_data == 0), 'Female': (sex_data == 1),
    'Age < 40': (age_data < 40), 'Age 40-59': (age_data >= 40) & (age_data < 60), 'Age >= 60': (age_data >= 60),
}
subgroup_rows = []
for g_name, mask in subgroups.items():
    row = {'Subgroup': g_name, 'N': int(mask.sum())}
    for name in top_models:
        if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
            row[f'AUC_{name}'] = round(roc_auc_score(y_test[mask], predictions[name][mask], sample_weight=w_test[mask]), 4)
        else:
            row[f'AUC_{name}'] = np.nan
    subgroup_rows.append(row)
subgroup_df = pd.DataFrame(subgroup_rows)
subgroup_df.to_csv(f'{RESULT_DIR}/3_subgroup_top3.csv', index=False)
print(subgroup_df.to_string(index=False))


# [PART 4] Verification of leakage of HE_obe information: Before/After comparison
df_obe = pd.read_csv(FULL_DATA_PATH_W_OBE)

from sklearn.model_selection import train_test_split as tts
df_tr_o, df_tmp_o = tts(df_obe, test_size=0.3, random_state=RANDOM_STATE, stratify=df_obe['MS'])
df_val_o, df_te_o = tts(df_tmp_o, test_size=(4010/6011), random_state=RANDOM_STATE, stratify=df_tmp_o['MS'])

def build_xyw(df, drop_obe):
    y = df['MS'].values; w = df['wt_pool'].values
    drop_cols = ['MS'] + meta_cols + (['HE_obe'] if drop_obe else [])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y, w

before_after_rows = []
for drop_obe, cond_label in [(True, 'Without HE_obe (Primary/Corrected)'), (False, 'With HE_obe (Original/Leaky)')]:
    X_tr, y_tr, w_tr = build_xyw(df_tr_o, drop_obe)
    X_te, y_te, w_te = build_xyw(df_te_o, drop_obe)
    sc = StandardScaler()
    X_tr_s = X_tr.copy(); X_tr_s[continuous_cols] = sc.fit_transform(X_tr[continuous_cols])
    X_te_s = X_te.copy(); X_te_s[continuous_cols] = sc.transform(X_te[continuous_cols])
    
    for name in top_models:
        if name in ML_NAMES:
            model = make_ml_models()[name]
            model.fit(X_tr_s, y_tr, sample_weight=w_tr)
            prob = model.predict_proba(X_te_s)[:, 1]
        else:
            model = train_dl_model(DL_CLASSES[name], X_tr_s, y_tr, w_tr, X_te_s, y_te, w_te, pos_weight=IMBALANCE_RATIO)
            prob = get_dl_probs(model, X_te_s)
        auc = roc_auc_score(y_te, prob, sample_weight=w_te)
        before_after_rows.append([cond_label, name, round(auc, 4)])

ba_df = pd.DataFrame(before_after_rows, columns=['Condition', 'Model', 'Weighted_AUC'])
ba_df.to_csv(f'{RESULT_DIR}/4_HEobe_leakage_before_after.csv', index=False)
print(ba_df.to_string(index=False))

import shap
X_tr_leak, y_tr_leak, w_tr_leak = build_xyw(df_tr_o, drop_obe=False)
X_te_leak, y_te_leak, w_te_leak = build_xyw(df_te_o, drop_obe=False)
sc_leak = StandardScaler()
X_tr_leak[continuous_cols] = sc_leak.fit_transform(X_tr_leak[continuous_cols])
X_te_leak[continuous_cols] = sc_leak.transform(X_te_leak[continuous_cols])
xgb_leak = XGBClassifier(n_estimators=580, max_depth=4, learning_rate=0.0101, subsample=0.923,
                          colsample_bytree=0.755, random_state=RANDOM_STATE, eval_metric='logloss',
                          scale_pos_weight=IMBALANCE_RATIO)
xgb_leak.fit(X_tr_leak, y_tr_leak, sample_weight=w_tr_leak)
explainer_leak = shap.TreeExplainer(xgb_leak)
shap_vals_leak = explainer_leak.shap_values(X_te_leak.sample(min(1000, len(X_te_leak)), random_state=RANDOM_STATE))
shap_vals_leak = shap_vals_leak[1] if isinstance(shap_vals_leak, list) else shap_vals_leak
mean_abs_shap = pd.Series(np.abs(shap_vals_leak).mean(axis=0), index=X_te_leak.columns).sort_values(ascending=False)
mean_abs_shap.head(10).to_frame('Mean_abs_SHAP').to_csv(f'{RESULT_DIR}/4_HEobe_shap_dominance.csv')
print(mean_abs_shap.head(10))


# [PART 5] Complex Sampling Design: Weighted/non-weighted comparison
weight_rows = []
for weighted in [True, False]:
    w_tr_use = w_train if weighted else np.ones_like(w_train)
    for name in top_models:
        if name in ML_NAMES:
            model = make_ml_models()[name]
            model.fit(X_train_scaled, y_train, sample_weight=w_tr_use)
            prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model = train_dl_model(DL_CLASSES[name], X_train_scaled, y_train, w_tr_use,
                                    X_val_scaled, y_val, w_val, pos_weight=IMBALANCE_RATIO)
            prob = get_dl_probs(model, X_test_scaled)
        auc_w = roc_auc_score(y_test, prob, sample_weight=w_test)     
        auc_u = roc_auc_score(y_test, prob)                            
        weight_rows.append(['Weighted Training' if weighted else 'Unweighted Training', name,
                             round(auc_w, 4), round(auc_u, 4)])
weight_df = pd.DataFrame(weight_rows, columns=['Training_Condition', 'Model', 'AUC_weighted_eval', 'AUC_unweighted_eval'])
weight_df.to_csv(f'{RESULT_DIR}/5_weighted_vs_unweighted.csv', index=False)
print(weight_df.to_string(index=False))


# [PART 6] Class imbalance correction 4-condition comparison (extended to Top-3 model)
# condition: A) Original Only, B) SMOTE Only, C) Class Weight Only, D) SMOTE + Class Weight
imbalance_rows = []
for name in top_models:
    is_dl = name in DL_CLASSES
    # A. Original Only 
    if is_dl:
        m_A = train_dl_model(DL_CLASSES[name], X_train_scaled, y_train, np.ones_like(w_train),
                              X_val_scaled, y_val, w_val, pos_weight=None)
        prob_A = get_dl_probs(m_A, X_test_scaled)
    else:
        m_A = make_ml_models()[name]; m_A.fit(X_train_scaled, y_train)
        prob_A = m_A.predict_proba(X_test_scaled)[:, 1]
    # B. SMOTE Only
    if is_dl:
        m_B = train_dl_model(DL_CLASSES[name], X_train_smote_scaled, y_train_smote, np.ones_like(y_train_smote, dtype=float),
                              X_val_scaled, y_val, w_val, pos_weight=None)
        prob_B = get_dl_probs(m_B, X_test_scaled)
    else:
        m_B = make_ml_models()[name]; m_B.fit(X_train_smote_scaled, y_train_smote)
        prob_B = m_B.predict_proba(X_test_scaled)[:, 1]
    # C. Class Weight Only 
    prob_C = predictions[name]
    # D. SMOTE + Class Weight
    if is_dl:
        m_D = train_dl_model(DL_CLASSES[name], X_train_smote_scaled, y_train_smote, np.ones_like(y_train_smote, dtype=float),
                              X_val_scaled, y_val, w_val, pos_weight=IMBALANCE_RATIO)
        prob_D = get_dl_probs(m_D, X_test_scaled)
    else:
        m_D = make_ml_models()[name]; m_D.fit(X_train_smote_scaled, y_train_smote)  
        prob_D = m_D.predict_proba(X_test_scaled)[:, 1]

    for cond_label, prob in [('A. Original Only', prob_A), ('B. SMOTE Only', prob_B),
                              ('C. Class Weight Only', prob_C), ('D. SMOTE + Class Weight', prob_D)]:
        pred = (prob >= 0.5).astype(int)
        imbalance_rows.append([name, cond_label,
                                round(roc_auc_score(y_test, prob, sample_weight=w_test), 4),
                                round(precision_score(y_test, pred, sample_weight=w_test), 4),
                                round(recall_score(y_test, pred, sample_weight=w_test), 4),
                                round(f1_score(y_test, pred, sample_weight=w_test), 4),
                                round(brier_score_loss(y_test, prob), 4)])
imbalance_df = pd.DataFrame(imbalance_rows, columns=['Model', 'Condition', 'AUC', 'Precision', 'Recall', 'F1', 'Brier'])
imbalance_df.to_csv(f'{RESULT_DIR}/6_class_imbalance_4conditions_top3.csv', index=False)
print(imbalance_df.to_string(index=False))


# [PART 7] Leave-One-Region-Out (LORO); Geographic Generalization Verification (Top-3 model)
df_full = pd.read_csv(FULL_DATA_PATH)
target_col, weight_col = 'MS', 'wt_pool'
loro_rows = []
regions = sorted(df_full['region'].unique())
for region in regions:
    train_r = df_full[df_full['region'] != region]
    test_r  = df_full[df_full['region'] == region]
    if test_r[target_col].nunique() < 2 or len(test_r) < 30:
        continue
    X_tr_r, y_tr_r, w_tr_r = build_xyw(train_r, drop_obe=True) if 'HE_obe' in df_full.columns else get_X_y_w(train_r)
    X_te_r, y_te_r, w_te_r = build_xyw(test_r, drop_obe=True) if 'HE_obe' in df_full.columns else get_X_y_w(test_r)
    X_tr_r = X_tr_r.drop(columns=['region'], errors='ignore'); X_te_r = X_te_r.drop(columns=['region'], errors='ignore')
    sc_r = StandardScaler()
    X_tr_r[continuous_cols] = sc_r.fit_transform(X_tr_r[continuous_cols])
    X_te_r[continuous_cols] = sc_r.transform(X_te_r[continuous_cols])
    for name in top_models:
        if name in ML_NAMES:
            model = make_ml_models()[name]; model.fit(X_tr_r, y_tr_r, sample_weight=w_tr_r)
            prob = model.predict_proba(X_te_r)[:, 1]
        else:
            model = train_dl_model(DL_CLASSES[name], X_tr_r, y_tr_r, w_tr_r, X_te_r, y_te_r, w_te_r, pos_weight=IMBALANCE_RATIO, patience=8)
            prob = get_dl_probs(model, X_te_r)
        auc = roc_auc_score(y_te_r, prob, sample_weight=w_te_r)
        loro_rows.append([region, name, len(test_r), round(auc, 4)])
loro_df = pd.DataFrame(loro_rows, columns=['Region', 'Model', 'N', 'AUC'])
loro_df.to_csv(f'{RESULT_DIR}/7_LORO_top3_models.csv', index=False)
loro_summary = loro_df.groupby('Model')['AUC'].agg(['mean', 'std']).reset_index()
loro_summary.to_csv(f'{RESULT_DIR}/7_LORO_summary_top3.csv', index=False)
print(loro_summary.to_string(index=False))


# [PART 8] Check source environment variables directly SHAP + multicollinearity (VIF) (Top-3 model)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def one_hot_groups(columns):
    groups = {}
    for c in columns:
        base = c.split('_')[0] if any(ch.isdigit() for ch in c.split('_')[-1]) else c
        
        import re
        m = re.match(r'^(.*)_[0-9]+\.0$', c)
        key = m.group(1) if m else c
        groups.setdefault(key, []).append(c)
    return groups

groups = one_hot_groups(X_test_scaled.columns)

def extract_positive_class_shap(sv):
    if isinstance(sv, list):
        return sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, -1]   
    return sv

shap_summary_rows = []
sample_idx = X_test_scaled.sample(min(300, len(X_test_scaled)), random_state=RANDOM_STATE).index
for name in top_models:
    X_bg = X_test_scaled.loc[sample_idx]
    if name in ['Random_Forest', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(ml_models[name])
        sv = explainer.shap_values(X_bg, check_additivity=False)
        sv = extract_positive_class_shap(sv)
    elif name == 'Logistic_Regression':
        explainer = shap.LinearExplainer(ml_models[name], X_train_scaled)
        sv = explainer.shap_values(X_bg)
        sv = extract_positive_class_shap(sv)
    else:  
        bg_small = X_train_scaled.sample(50, random_state=RANDOM_STATE)
        if name in DL_CLASSES:
            f = lambda x: get_dl_probs(dl_models[name], pd.DataFrame(x, columns=X_bg.columns))
        else:
            f = lambda x: ml_models[name].predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(f, bg_small)
        X_bg = X_bg.sample(min(200, len(X_bg)), random_state=RANDOM_STATE)
        sv = explainer.shap_values(X_bg, nsamples=100)
        sv = extract_positive_class_shap(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    per_col = pd.Series(mean_abs, index=X_bg.columns)
    grouped = pd.Series({g: per_col[cols].sum() for g, cols in groups.items()}).sort_values(ascending=False)
    for var, val in grouped.items():
        shap_summary_rows.append([name, var, round(val, 4)])

shap_summary_df = pd.DataFrame(shap_summary_rows, columns=['Model', 'Variable', 'Mean_abs_SHAP'])
shap_summary_df.to_csv(f'{RESULT_DIR}/8_shap_original_vars_top3.csv', index=False)
print(shap_summary_df.groupby('Model').head(10).to_string(index=False))

# VIF (7 original environmental variables + overall predictors)
vif_env = pd.DataFrame({'Variable': continuous_cols[1:],  
                         'VIF': [variance_inflation_factor(X_test_scaled[continuous_cols[1:]].values, i) for i in range(len(continuous_cols[1:]))]})
vif_env.to_csv(f'{RESULT_DIR}/8_vif_check.csv', index=False)
print(vif_env.to_string(index=False))


# [PART 9] Master Sensitivity Summary
# PCA vs Original, With/Without HE_obe, Four Unbalanced Correction, Weighted vs Unweighted, Single Split vs 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
X_full_train = pd.concat([X_train_scaled, X_val_scaled], axis=0).reset_index(drop=True)
y_full_train = np.concatenate([y_train, y_val])
w_full_train = np.concatenate([w_train, w_val])
for name in top_models:
    fold_aucs = []
    for tr_idx, te_idx in skf.split(X_full_train, y_full_train):
        X_tr_k, X_te_k = X_full_train.iloc[tr_idx], X_full_train.iloc[te_idx]
        y_tr_k, y_te_k = y_full_train[tr_idx], y_full_train[te_idx]
        w_tr_k, w_te_k = w_full_train[tr_idx], w_full_train[te_idx]
        if name in ML_NAMES:
            model = make_ml_models()[name]; model.fit(X_tr_k, y_tr_k, sample_weight=w_tr_k)
            prob = model.predict_proba(X_te_k)[:, 1]
        else:
            model = train_dl_model(DL_CLASSES[name], X_tr_k, y_tr_k, w_tr_k, X_te_k, y_te_k, w_te_k, pos_weight=IMBALANCE_RATIO, patience=8)
            prob = get_dl_probs(model, X_te_k)
        fold_aucs.append(roc_auc_score(y_te_k, prob, sample_weight=w_te_k))
    single_split_auc = roc_auc_score(y_test, predictions[name], sample_weight=w_test)
    cv_rows.append([name, single_split_auc, np.mean(fold_aucs), np.std(fold_aucs)])
cv_df = pd.DataFrame(cv_rows, columns=['Model', 'Single_Split_AUC', 'CV5_Mean_AUC', 'CV5_Std_AUC'])
cv_df.to_csv(f'{RESULT_DIR}/9_5fold_cv_vs_single_split.csv', index=False)

sensitivity_master = pd.DataFrame({
    'Sensitivity_Axis': ['Predictor set (With/Without HE_obe)', 'Sample weighting (Weighted/Unweighted)',
                          'Class imbalance correction (4 conditions)', 'Data split (Single split vs 5-Fold CV)',
                          'Geographic split (Random vs LORO)'],
    'Output_File': ['4_HEobe_leakage_before_after.csv', '5_weighted_vs_unweighted.csv',
                     '6_class_imbalance_4conditions_top3.csv', '9_5fold_cv_vs_single_split.csv',
                     '7_LORO_summary_top3.csv'],
})
sensitivity_master.to_csv(f'{RESULT_DIR}/9_sensitivity_summary_master.csv', index=False)
print(cv_df.to_string(index=False))


# [PART 10] Quantifying the contribution of environmental variables (especially Green Space) -> Causal Expression Mitigation Evidence
green_ratio_rows = []
for name in top_models:
    sub = shap_summary_df[shap_summary_df['Model'] == name].set_index('Variable')['Mean_abs_SHAP']
    if 'green_1' in sub.index:
        top1_val = sub.iloc[0]
        green_val = sub['green_1']
        green_rank = int((sub.rank(ascending=False))['green_1'])
        green_ratio_rows.append([name, sub.index[0], round(top1_val, 4), round(green_val, 4),
                                  green_rank, round(green_val / top1_val * 100, 2)])
green_ratio_df = pd.DataFrame(green_ratio_rows, columns=['Model', 'Top1_Variable', 'Top1_SHAP',
                                                          'green_1_SHAP', 'green_1_Rank', 'green_1_pct_of_top1'])
green_ratio_df.to_csv(f'{RESULT_DIR}/10_green1_contribution_ratio.csv', index=False)
print(green_ratio_df.to_string(index=False))


# [PART 11] Public Health Utilization: Precision-Recall / Threshold Sweep (Top-3 model)
plt.figure(figsize=(7, 7))
threshold_rows = []
for name in top_models:
    prob = predictions[name]
    prec, rec, thr = precision_recall_curve(y_test, prob, sample_weight=w_test)
    auprc = average_precision_score(y_test, prob, sample_weight=w_test)
    plt.plot(rec, prec, label=f'{name} (AUPRC={auprc:.4f})')
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_t = (prob >= t).astype(int)
        p_t = precision_score(y_test, pred_t, sample_weight=w_test, zero_division=0)
        r_t = recall_score(y_test, pred_t, sample_weight=w_test, zero_division=0)
        nns = round(1 / p_t, 1) if p_t > 0 else np.nan  
        threshold_rows.append([name, t, round(p_t, 4), round(r_t, 4), nns])
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve - Top 3 Models')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f'{RESULT_DIR}/11_pr_curve_top3.png', dpi=300)
plt.show()
threshold_df = pd.DataFrame(threshold_rows, columns=['Model', 'Threshold', 'Precision', 'Recall', 'Approx_NNS'])
threshold_df.to_csv(f'{RESULT_DIR}/11_threshold_sweep_top3.csv', index=False)
print(threshold_df.to_string(index=False))


