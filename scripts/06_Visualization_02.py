"""
Generation files (4 CSV):
  1) 13_pca_vs_original_top3.csv        - Original 7 Environmental Variables vs PCA 3 Component AUC Comparison
  2) 13_group_shap_by_cluster.csv       - Correlation Cluster Unit Group SHAP
  3) 13_ridge_vs_plain_coefficients.csv - Ridge(L2) vs Nonnormalized Logistic Factor Comparison
  4) 13_pm10_pm25_leaveoneout.csv       - PM10-PM2.5 pair: leave-one-out
"""

# Library install, if the session was re-started.
# !pip install -q shap xgboost

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Google Drive mount
# from google.colab import drive
# drive.mount('/content/drive')

DATA_PATH = "/content/drive/MyDrive/MetS_Experiment/data/knhanes_encoded_final.csv"
OUT_DIR = "/content/drive/MyDrive/MetS_Experiment/data/results/figures"

RANDOM_SEED = 42
MAX_EPOCHS = 200
PATIENCE = 15
SHAP_MAX_SAMPLES = 300

TARGET = "MS"
ENV_VARS = ["CO", "NO2", "O3", "PM10", "PM2_5", "SO2", "green_1"]
CONTINUOUS_VARS = ["age"] + ENV_VARS
META_COLS = ["year", "wt_itvex", "kstrata", "psu", "wt_pool"]
WEIGHT_COL = "wt_pool"
IMBALANCE_RATIO = 2.54


class DeepMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def get_dl_probs(model, X):
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.FloatTensor(X))).numpy().flatten()


def train_dl_model(model_class, X_tr, y_tr, w_tr, X_v, y_v, w_v,
                    pos_weight=None, patience=PATIENCE, max_epochs=MAX_EPOCHS, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_v = np.asarray(X_v, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)
    y_v = np.asarray(y_v, dtype=np.float32)
    w_tr = np.asarray(w_tr, dtype=np.float32)
    w_v = np.asarray(w_v, dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).unsqueeze(1),
                      torch.FloatTensor(w_tr).unsqueeze(1)),
        batch_size=256, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v).unsqueeze(1),
                      torch.FloatTensor(w_v).unsqueeze(1)),
        batch_size=256, shuffle=False,
    )

    model = model_class(X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = (nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), reduction="none")
                 if pos_weight else nn.BCEWithLogitsLoss(reduction="none"))

    best_val_loss, patience_counter, best_weights = float("inf"), 0, None
    for epoch in range(max_epochs):
        model.train()
        for b_X, b_y, b_w in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            (loss * b_w).mean().backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_X, b_y, b_w in val_loader:
                val_loss += (criterion(model(b_X), b_y) * b_w).mean().item()

        if val_loss < best_val_loss:
            best_val_loss, patience_counter, best_weights = val_loss, 0, model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def make_ml_models(seed=RANDOM_SEED):
    from xgboost import XGBClassifier
    return {
        "Random_Forest": RandomForestClassifier(
            n_estimators=380, max_depth=8, min_samples_split=6, max_features="sqrt",
            random_state=seed, class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=580, max_depth=4, learning_rate=0.0101,
            subsample=0.923, colsample_bytree=0.755,
            random_state=seed, eval_metric="logloss", scale_pos_weight=IMBALANCE_RATIO,
        ),
    }


MODEL_NAMES = ["Random_Forest", "XGBoost", "DeepMLP"]


def fit_and_predict(model_name, X_tr, y_tr, w_tr, X_v, y_v, w_v, X_te, seed=RANDOM_SEED, patience=PATIENCE):
    if model_name == "DeepMLP":
        model = train_dl_model(DeepMLP, np.asarray(X_tr), y_tr, w_tr,
                                np.asarray(X_v), y_v, w_v,
                                pos_weight=IMBALANCE_RATIO, patience=patience, seed=seed)
        proba_te = get_dl_probs(model, np.asarray(X_te))
    else:
        model = make_ml_models(seed)[model_name]
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        proba_te = model.predict_proba(X_te)[:, 1]
    return model, proba_te


# Common Data Util
def load_data(data_path):
    df = pd.read_csv(data_path).dropna().reset_index(drop=True)
    missing = [c for c in ENV_VARS + [TARGET, WEIGHT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Necessary column missing on data: {missing}")
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS + [TARGET]]


def split_train_val_test(df, seed=RANDOM_SEED):
    y = df[TARGET]
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=y, random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=(4011 / 6011), stratify=temp_df[TARGET], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def get_mean_abs_shap(model, model_name, X_background, X_eval, feature_names, seed=RANDOM_SEED):
    import shap

    def extract_pos(sv):
        if isinstance(sv, list):
            return sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            return sv[:, :, -1]
        return sv

    if model_name in ("Random_Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_eval, check_additivity=False)
        sv = extract_pos(sv)
    else:  
        bg_small = X_background.sample(min(50, len(X_background)), random_state=seed)
        f = lambda x: get_dl_probs(model, x)
        explainer = shap.KernelExplainer(f, bg_small)
        X_eval = X_eval.sample(min(200, len(X_eval)), random_state=seed)
        sv = explainer.shap_values(X_eval, nsamples=100)
        sv = extract_pos(sv)

    mean_abs = np.abs(sv).mean(axis=0)
    return dict(zip(feature_names, mean_abs))


# Analysis 1 - PCA vs. original variable AUC
def analysis1_pca_vs_original(df, out_dir, seed=RANDOM_SEED):
    print("\n[Analysis 1] PCA vs. original variable AUC")
    feature_cols = get_feature_cols(df)
    non_env_cols = [c for c in feature_cols if c not in ENV_VARS]
    train_df, val_df, test_df = split_train_val_test(df, seed)

    y_train = train_df[TARGET].astype(int).values
    y_val = val_df[TARGET].astype(int).values
    y_test = test_df[TARGET].astype(int).values
    w_train = train_df[WEIGHT_COL].values
    w_val = val_df[WEIGHT_COL].values
    w_test = test_df[WEIGHT_COL].values

    age_scaler = StandardScaler().fit(train_df[["age"]])

    rows = []
    for feature_set_name, n_components in [("Original_7Vars", None), ("PCA_3Components", 3)]:
        env_scaler = StandardScaler().fit(train_df[ENV_VARS])
        env_train = env_scaler.transform(train_df[ENV_VARS])
        env_val = env_scaler.transform(val_df[ENV_VARS])
        env_test = env_scaler.transform(test_df[ENV_VARS])

        if n_components is not None:
            pca = PCA(n_components=n_components, random_state=seed).fit(env_train)
            env_train, env_val, env_test = (pca.transform(env_train),
                                             pca.transform(env_val),
                                             pca.transform(env_test))
            env_cols = [f"PC{i+1}" for i in range(n_components)]
            explained_var = float(pca.explained_variance_ratio_.sum())
        else:
            env_cols = ENV_VARS
            explained_var = 1.0

        def build_X(base_df, env_arr):
            X_env = pd.DataFrame(env_arr, columns=env_cols, index=base_df.index)
            X_other = base_df[non_env_cols].copy()
            X_other["age"] = age_scaler.transform(base_df[["age"]])
            return pd.concat([X_other, X_env], axis=1)

        X_train = build_X(train_df, env_train)
        X_val = build_X(val_df, env_val)
        X_test = build_X(test_df, env_test)
        cols = list(X_train.columns)

        for model_name in MODEL_NAMES:
            _, proba = fit_and_predict(model_name, X_train, y_train, w_train,
                                        X_val, y_val, w_val, X_test, seed=seed)
            auc = roc_auc_score(y_test, proba, sample_weight=w_test)
            rows.append({
                "Model": model_name,
                "Feature_Set": feature_set_name,
                "N_Features": len(cols),
                "Cumulative_Explained_Variance": round(explained_var, 4),
                "AUC": round(auc, 4),
            })
            print(f"  {model_name:15s} | {feature_set_name:18s} | "
                  f"N_features={len(cols):2d} | AUC={auc:.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "13_pca_vs_original_top3.csv")
    out.to_csv(path, index=False)
    print(f"  saved: {path}")
    return out


# Analysis 2 - Correlation Cluster Unit Group SHAP
def find_correlation_clusters(df, threshold=0.7):
    corr = df[ENV_VARS].corr().abs()
    n = len(ENV_VARS)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if corr.iloc[i, j] >= threshold:
                union(i, j)

    clusters = {}
    for i, v in enumerate(ENV_VARS):
        clusters.setdefault(find(i), []).append(v)
    return list(clusters.values())


def analysis2_group_shap(df, out_dir, seed=RANDOM_SEED, corr_threshold=0.7):
    print("\n[Analysis 2] Correlation Cluster Unit Group SHAP")
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split_train_val_test(df, seed)

    scaler = StandardScaler().fit(train_df[CONTINUOUS_VARS])
    X_train = train_df[feature_cols].copy(); X_train[CONTINUOUS_VARS] = scaler.transform(train_df[CONTINUOUS_VARS])
    X_test = test_df[feature_cols].copy(); X_test[CONTINUOUS_VARS] = scaler.transform(test_df[CONTINUOUS_VARS])
    y_train = train_df[TARGET].astype(int).values
    w_train = train_df[WEIGHT_COL].values

    clusters = find_correlation_clusters(df, threshold=corr_threshold)
    print(f"  Found correlation cluster(|r|>={corr_threshold}): {clusters}")

    X_eval = X_test.sample(min(SHAP_MAX_SAMPLES, len(X_test)), random_state=seed)

    rows = []
    for model_name in ["Random_Forest", "XGBoost"]:  
        model = make_ml_models(seed)[model_name]
        model.fit(X_train, y_train, sample_weight=w_train)
        shap_lookup = get_mean_abs_shap(model, model_name, X_train, X_eval, feature_cols, seed)

        for cluster in clusters:
            group_sum = sum(shap_lookup[v] for v in cluster)
            rows.append({
                "Model": model_name,
                "Cluster_Variables": "+".join(cluster),
                "N_Vars_in_Cluster": len(cluster),
                "Sum_Mean_Abs_SHAP": round(group_sum, 4),
                "Individual_SHAP": "; ".join(f"{v}={shap_lookup[v]:.4f}" for v in cluster),
            })
            print(f"  {model_name:15s} | cluster={'+'.join(cluster):20s} | "
                  f"group SHAP sum={group_sum:.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "13_group_shap_by_cluster.csv")
    out.to_csv(path, index=False)
    print(f"  saved: {path}")
    return out


# Analysis 3 - Ridge(L2) vs Nonnormalized Logistic Factor Comparison
def analysis3_ridge_coefficients(df, out_dir, seed=RANDOM_SEED):
    print("\n[Analysis 3] Ridge(L2) vs Nonnormalized Logistic Factor Comparison")
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split_train_val_test(df, seed)

    scaler = StandardScaler().fit(train_df[feature_cols])
    X_train_scaled = scaler.transform(train_df[feature_cols])
    y_train = train_df[TARGET].astype(int).values
    w_train = train_df[WEIGHT_COL].values

    ridge_cv = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 20), penalty="l2", solver="lbfgs",
        cv=5, max_iter=3000, scoring="roc_auc", random_state=seed,
    )
    ridge_cv.fit(X_train_scaled, y_train, sample_weight=w_train)
    ridge_coefs = ridge_cv.coef_[0]
    selected_C = float(ridge_cv.C_[0])

    plain = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=3000, random_state=seed)
    plain.fit(X_train_scaled, y_train, sample_weight=w_train)
    plain_coefs = plain.coef_[0]

    rows = []
    for i, col in enumerate(feature_cols):
        rows.append({
            "Variable": col,
            "Is_Environmental": col in ENV_VARS,
            "Ridge_C_selected": round(selected_C, 5),
            "Ridge_Coefficient": round(ridge_coefs[i], 4),
            "Plain_Logistic_Coefficient": round(plain_coefs[i], 4),
            "Shrinkage_Ratio(Ridge/Plain)": (
                round(ridge_coefs[i] / plain_coefs[i], 3) if abs(plain_coefs[i]) > 1e-6 else np.nan
            ),
        })
    out = pd.DataFrame(rows).sort_values("Is_Environmental", ascending=False).reset_index(drop=True)
    path = os.path.join(out_dir, "13_ridge_vs_plain_coefficients.csv")
    out.to_csv(path, index=False)
    print(out[out["Is_Environmental"]].to_string(index=False))
    print(f"  saved: {path}")
    return out


# Analysis 4 - PM10 / PM2.5 leave-one-out
def analysis4_pm_leaveoneout(df, out_dir, seed=RANDOM_SEED):
    print("\n[Analysis 4] PM10-PM2.5 pair leave-one-out Analysis")
    feature_cols = get_feature_cols(df)
    other_cols = [c for c in feature_cols if c not in ("PM10", "PM2_5")]
    train_df, val_df, test_df = split_train_val_test(df, seed)

    conditions = {
        "Both_PM10_PM2.5": feature_cols,
        "PM10_only": other_cols + ["PM10"],
        "PM2.5_only": other_cols + ["PM2_5"],
    }

    y_train = train_df[TARGET].astype(int).values
    y_val = val_df[TARGET].astype(int).values
    y_test = test_df[TARGET].astype(int).values
    w_train = train_df[WEIGHT_COL].values
    w_val = val_df[WEIGHT_COL].values
    w_test = test_df[WEIGHT_COL].values

    rows = []
    for cond_name, cols in conditions.items():
        cont_cols = [c for c in CONTINUOUS_VARS if c in cols]
        scaler = StandardScaler().fit(train_df[cont_cols])
        X_train = train_df[cols].copy(); X_train[cont_cols] = scaler.transform(train_df[cont_cols])
        X_val = val_df[cols].copy(); X_val[cont_cols] = scaler.transform(val_df[cont_cols])
        X_test = test_df[cols].copy(); X_test[cont_cols] = scaler.transform(test_df[cont_cols])

        for model_name in MODEL_NAMES:
            model, proba = fit_and_predict(model_name, X_train, y_train, w_train,
                                            X_val, y_val, w_val, X_test, seed=seed)
            auc = roc_auc_score(y_test, proba, sample_weight=w_test)

            retained_shap_val = np.nan
            if model_name in ("Random_Forest", "XGBoost"):
                X_eval = X_test.sample(min(SHAP_MAX_SAMPLES, len(X_test)), random_state=seed)
                shap_lookup = get_mean_abs_shap(model, model_name, X_train, X_eval, cols, seed)
                if "PM10" in cols and "PM2_5" in cols:
                    retained_shap_val = shap_lookup["PM10"] + shap_lookup["PM2_5"]
                elif "PM10" in cols:
                    retained_shap_val = shap_lookup["PM10"]
                elif "PM2_5" in cols:
                    retained_shap_val = shap_lookup["PM2_5"]

            rows.append({
                "Condition": cond_name,
                "Model": model_name,
                "N_Features": len(cols),
                "AUC": round(auc, 4),
                "PM_Mean_Abs_SHAP": (round(retained_shap_val, 4)
                                     if not np.isnan(retained_shap_val) else "n/a (DeepMLP)"),
            })
            print(f"  {cond_name:16s} | {model_name:15s} | AUC={auc:.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "13_pm10_pm25_leaveoneout.csv")
    out.to_csv(path, index=False)
    print(f"  saved: {path}")
    return out


def run_all(data_path, out_dir, seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Reading data from : {data_path}")
    print(f"Writing results to: {out_dir}")
    if not os.path.isfile(data_path):
        print(f"  !! DATA_PATH does not exist: {data_path}\n"
              f"     Double-check the path (and that Google Drive is mounted, if used).")
        return

    df = load_data(data_path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns. "
          f"MS positive rate = {df[TARGET].mean():.3%}")

    analysis1_pca_vs_original(df, out_dir, seed)
    analysis2_group_shap(df, out_dir, seed)
    analysis3_ridge_coefficients(df, out_dir, seed)
    analysis4_pm_leaveoneout(df, out_dir, seed)

    print("\nAll 4 analyses complete.")


def main():
    data_path, out_dir = DATA_PATH, OUT_DIR
    argv = sys.argv[1:]
    if argv:
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--data-path", default=DATA_PATH)
        parser.add_argument("--out-dir", default=OUT_DIR)
        args, _unknown = parser.parse_known_args(argv)
        data_path, out_dir = args.data_path, args.out_dir
    run_all(data_path, out_dir)


if __name__ == "__main__":
    main()