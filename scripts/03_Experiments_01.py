import os
import re
import joblib
import numpy as np
import pandas as pd
import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')


# Google Drive load
from google.colab import drive
drive.mount('/content/drive')


BASE       = '/content/drive/MyDrive/MetS_Experiment/'
DATA_DIR   = f'{BASE}/data/'
MODEL_DIR  = f'{BASE}/models'
RESULT_DIR = f'{BASE}/results'
os.makedirs(RESULT_DIR, exist_ok=True)

meta_cols       = ['year', 'wt_itvex', 'wt_pool', 'kstrata', 'psu']
continuous_cols = ['age', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 'green_1']

def get_X_y_w(df, has_weight=True):
    y = df['MS'].values
    w = df['wt_pool'].values if has_weight else np.ones(len(df))
    cols_to_drop = ['MS'] + [c for c in meta_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X, y, w

test_df = pd.read_csv(f'{DATA_DIR}/test_set.csv')
X_test, y_test, w_test = get_X_y_w(test_df, has_weight=True)

scaler = joblib.load(f'{MODEL_DIR}/standard_scaler.pkl')
X_test_scaled = X_test.copy()
X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"테스트셋 로드 완료: {len(X_test_scaled)}명, 변수 {X_test_scaled.shape[1]}개")


# Deep learning model class re-define and saved 8-models (No re-training)
class BasicMLP(nn.Module):
    def __init__(self, in_dim):
        super(BasicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

class DeepMLP(nn.Module):
    def __init__(self, in_dim):
        super(DeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.layer(x) + x)

class ResNetMLP(nn.Module):
    def __init__(self, in_dim):
        super(ResNetMLP, self).__init__()
        self.fc_in = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.res_blocks = nn.Sequential(ResBlock(64), ResBlock(64), ResBlock(64))
        self.fc_out = nn.Linear(64, 1)
    def forward(self, x): return self.fc_out(self.res_blocks(self.fc_in(x)))

def dl_predict_proba(model, X_numpy):
    model.eval()
    with torch.no_grad():
        t = torch.FloatTensor(X_numpy)
        return torch.sigmoid(model(t)).numpy().flatten()

in_dim = X_test_scaled.shape[1]
dl_model_classes = {'BasicMLP': BasicMLP, 'DeepMLP': DeepMLP, 'ResNetMLP': ResNetMLP}
dl_models = {}
for name, cls in dl_model_classes.items():
    m = cls(in_dim)
    m.load_state_dict(torch.load(f'{MODEL_DIR}/{name}.pth', map_location='cpu'))
    m.eval()
    dl_models[name] = m

ml_model_names = ['Logistic_Regression', 'Random_Forest', 'XGBoost', 'LightGBM', 'SVM']
ml_models = {name: joblib.load(f'{MODEL_DIR}/{name}.pkl') for name in ml_model_names}

all_model_names = ml_model_names + list(dl_model_classes.keys())
print("All trained models load completed (No re-training):", all_model_names)

# Predict function by models
def get_predict_fn(name):
    if name in ml_models:
        model = ml_models[name]
        return lambda X: model.predict_proba(X)[:, 1]
    else:
        model = dl_models[name]
        return lambda X: dl_predict_proba(model, X.values if hasattr(X, 'values') else X)

predict_fns = {name: get_predict_fn(name) for name in all_model_names}


# One-Hot group search
def infer_groups(columns):
    pattern = re.compile(r'^(.*)_(\d+\.\d+)$')
    groups = {}
    for col in columns:
        m = pattern.match(col)
        base = m.group(1) if m else col
        groups.setdefault(base, []).append(col)
    return groups

feature_names = X_test_scaled.columns.tolist()
feature_groups = infer_groups(feature_names)
group_order = list(feature_groups.keys())
multi_dummy_groups = {k: v for k, v in feature_groups.items() if len(v) > 1}

print(f"\nTotal {len(group_order)} variable groups, "
      f"Multi dummy groups: {len(multi_dummy_groups)} ea)")


# Test-set prediction and performance indication
def calc_auc_ci(y_true, y_prob, w_true, n_bootstraps=1000, seed=42):
    scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx], sample_weight=w_true[idx]))
    scores.sort()
    return scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores))]

predictions = {name: fn(X_test_scaled) for name, fn in predict_fns.items()}

table8_rows = []
for name, prob in predictions.items():
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, prob, sample_weight=w_test)
    ci_lo, ci_hi = calc_auc_ci(y_test, prob, w_test)
    auprc = average_precision_score(y_test, prob, sample_weight=w_test)
    acc = accuracy_score(y_test, pred, sample_weight=w_test)
    prec = precision_score(y_test, pred, sample_weight=w_test)
    rec = recall_score(y_test, pred, sample_weight=w_test)
    f1 = f1_score(y_test, pred, sample_weight=w_test)
    table8_rows.append([name, auc, f"[{ci_lo:.4f}-{ci_hi:.4f}]", auprc, f1, acc, prec, rec])

table8_df = pd.DataFrame(table8_rows, columns=[
    'Model', 'AUC-ROC', 'AUC_95%_CI', 'AUPRC', 'F1-Score', 'Accuracy', 'Precision', 'Recall'
]).sort_values('AUC-ROC', ascending=False)
table8_df.to_csv(f'{RESULT_DIR}/weighted_performance.csv', index=False)
print("\nWeighted Performance result saved")
print(table8_df.round(4).to_string(index=False))


# Permutation Importance (by group) - 8-models
def grouped_permutation_importance(predict_fn, X, y, w, groups, group_order,
                                    n_repeats=10, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline_pred = predict_fn(X)
    baseline_score = roc_auc_score(y, baseline_pred, sample_weight=w)

    importances = {}
    for gname in group_order:
        cols = groups[gname]
        col_idx = [X.columns.get_loc(c) for c in cols]
        drop_scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_order = rng.permutation(len(X))
            X_perm.iloc[:, col_idx] = X.iloc[perm_order, col_idx].values
            perm_pred = predict_fn(X_perm)
            perm_score = roc_auc_score(y, perm_pred, sample_weight=w)
            drop_scores.append(baseline_score - perm_score)  
        importances[gname] = np.mean(drop_scores)
    return pd.Series(importances, name='Permutation_Importance')

perm_importance_by_model = {}
for name in all_model_names:
    print(f"[Permutation Importance] {name} calculating ...")
    perm_importance_by_model[name] = grouped_permutation_importance(
        predict_fns[name], X_test_scaled, y_test, w_test,
        feature_groups, group_order, n_repeats=10, random_state=42)

perm_df = pd.DataFrame(perm_importance_by_model)
perm_df['Mean_Across_Models'] = perm_df.mean(axis=1)
perm_df = perm_df.sort_values('Mean_Across_Models', ascending=False)
perm_df.to_csv(f'{RESULT_DIR}/permutation_importance_grouped.csv')
print("\nPermutation Importance result")
print(perm_df.round(4).to_string())


# SHAP calculation (8-models)
def extract_positive_class_shap(sv):
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    elif sv.ndim != 2:
        raise ValueError(f"Unexpected SHAP value type: shape={sv.shape}")
    return sv

def combine_one_hot(shap_matrix, feat_names, groups):
    col_idx = {c: i for i, c in enumerate(feat_names)}
    out = {}
    for gname, cols in groups.items():
        idxs = [col_idx[c] for c in cols if c in col_idx]
        out[gname] = shap_matrix[:, idxs].sum(axis=1)
    return pd.DataFrame(out)

N_BACKGROUND = 50
N_EVAL = 200  
background = shap.sample(X_test_scaled, N_BACKGROUND, random_state=42)
eval_sample = X_test_scaled.sample(n=min(N_EVAL, len(X_test_scaled)), random_state=42)

grouped_shap_values = {}       
grouped_importance_by_model = {}  

# TreeExplainer: (Random Forest, XGBoost, LightGBM)
tree_models = {'Random_Forest': ml_models['Random_Forest'],
               'XGBoost': ml_models['XGBoost'],
               'LightGBM': ml_models['LightGBM']}
for name, model in tree_models.items():
    print(f"\n[TreeExplainer] {name} calculating...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(eval_sample, check_additivity=False)
    sv = extract_positive_class_shap(sv)
    grouped = combine_one_hot(sv, feature_names, feature_groups)
    grouped_shap_values[name] = grouped
    grouped_importance_by_model[name] = grouped.abs().mean(axis=0)

# Logistic Regression: LinearExplainer
print("\n[LinearExplainer] Logistic_Regression calculating...")
lr_explainer = shap.LinearExplainer(ml_models['Logistic_Regression'], X_test_scaled)
sv_lr = extract_positive_class_shap(lr_explainer.shap_values(eval_sample))
grouped_lr = combine_one_hot(sv_lr, feature_names, feature_groups)
grouped_shap_values['Logistic_Regression'] = grouped_lr
grouped_importance_by_model['Logistic_Regression'] = grouped_lr.abs().mean(axis=0)

# SVM: KernelExplainer
print("\n[KernelExplainer] SVM calculating...")
svm_explainer = shap.KernelExplainer(
    lambda x: ml_models['SVM'].predict_proba(x)[:, 1], background.values)
sv_svm = extract_positive_class_shap(svm_explainer.shap_values(eval_sample.values, nsamples=100))
grouped_svm = combine_one_hot(sv_svm, feature_names, feature_groups)
grouped_shap_values['SVM'] = grouped_svm
grouped_importance_by_model['SVM'] = grouped_svm.abs().mean(axis=0)

# 3-Deep Learning models: KernelExplainer
for name, model in dl_models.items():
    print(f"\n[KernelExplainer] {name} calculating... ")
    predict_fn = lambda x, m=model: dl_predict_proba(m, x)
    explainer = shap.KernelExplainer(predict_fn, background.values)
    sv = extract_positive_class_shap(explainer.shap_values(eval_sample.values, nsamples=100))
    grouped = combine_one_hot(sv, feature_names, feature_groups)
    grouped_shap_values[name] = grouped
    grouped_importance_by_model[name] = grouped.abs().mean(axis=0)

shap_summary_df = pd.DataFrame(grouped_importance_by_model)
shap_summary_df['Mean_Across_Models'] = shap_summary_df.mean(axis=1)
shap_summary_df = shap_summary_df.sort_values('Mean_Across_Models', ascending=False)
shap_summary_df.to_csv(f'{RESULT_DIR}/13_shap_importance_grouped_all_models.csv')
print("\nSHAP importance")
print(shap_summary_df.round(4).to_string())


# SHAP + Permutation Importance table/figure
fig3_df = pd.DataFrame({
    'Mean_abs_SHAP': shap_summary_df['Mean_Across_Models'],
    'Permutation_Importance': perm_df['Mean_Across_Models']
}).reindex(group_order)
fig3_df['SHAP_Rank'] = fig3_df['Mean_abs_SHAP'].rank(ascending=False).astype(int)
fig3_df['Perm_Rank'] = fig3_df['Permutation_Importance'].rank(ascending=False).astype(int)
fig3_df = fig3_df.sort_values('Mean_abs_SHAP', ascending=False)
fig3_df.to_csv(f'{RESULT_DIR}/14_figure3_shap_and_permutation_combined.csv')
print("\n[Figure 3: SHAP + Permutation Importance]")
print(fig3_df.round(4).to_string())

plt.figure(figsize=(8, 9))
colors = ['#C0504D' if g in multi_dummy_groups else '#4472C4' for g in fig3_df.index]
plt.barh(fig3_df.index[::-1], fig3_df['Mean_abs_SHAP'][::-1], color=colors[::-1])
plt.xlabel('Mean |SHAP Value| (8-Model Average, One-Hot Group Consolidation)')
plt.title('Figure 3: Variable Importance (Reviewer-Revised Pipeline)')
plt.tight_layout()
plt.savefig(f'{RESULT_DIR}/14_figure3_reproduced.png', dpi=300)
plt.show()


# BasicMLP SHAP Summary(Beeswarm) Plot
def compute_group_representative_value(X, groups, group_order):
    rep = {}
    for gname in group_order:
        cols = groups[gname]
        if len(cols) == 1:
            rep[gname] = X[cols[0]].values
        else:
            sub = X[cols].values
            level = np.zeros(len(X))
            active_idx = np.argmax(sub, axis=1)
            has_active = sub.max(axis=1) > 0.5
            level[has_active] = active_idx[has_active] + 1
            rep[gname] = level
    return pd.DataFrame(rep, index=X.index)

group_repr_values = compute_group_representative_value(eval_sample, feature_groups, group_order)

best_model_name = 'BasicMLP'
grouped_shap_best = grouped_shap_values[best_model_name][group_order]

explanation = shap.Explanation(
    values=grouped_shap_best.values,
    data=group_repr_values[group_order].values,
    feature_names=group_order
)

plt.figure(figsize=(9, 10))
shap.plots.beeswarm(explanation, max_display=len(group_order), show=False)
plt.title(f'Figure 4 : SHAP Summary Plot - {best_model_name} (Reviewer-Revised Pipeline)')
plt.tight_layout()
plt.savefig(f'{RESULT_DIR}/15_figure4_reproduced_beeswarm.png', dpi=300)
plt.show()



