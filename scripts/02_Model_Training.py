import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, accuracy_score,
                             confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Preprocessed data load
df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/knhanes_encoded_final.csv')

meta_cols = ['year', 'wt_itvex', 'wt_pool', 'kstrata', 'psu']

# Data split (Train 70%, Validation 10%, Test 20%)
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['MS'])
df_val, df_test = train_test_split(df_temp, test_size=(4010/6011), random_state=42, stratify=df_temp['MS'])

print(f"Data split completed - Train: {len(df_train)} people, Validation: {len(df_val)} people, Test: {len(df_test)} people")

# SMOTE applicatation
X_train = df_train.drop(columns=['MS'] + meta_cols)
y_train = df_train['MS']

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f" Train-set after SMOTE applicatation: {len(X_train_smote)} people (MS positive: {sum(y_train_smote==1)} people, Ratio: {y_train_smote.mean():.1%})")

df_train.to_csv('train_original.csv', index=False, encoding='utf-8-sig')

train_smote_df = pd.concat([X_train_smote, y_train_smote], axis=1)
train_smote_df.to_csv('train_smote.csv', index=False, encoding='utf-8-sig')

df_val.to_csv('validation_set.csv', index=False, encoding='utf-8-sig')
df_test.to_csv('test_set.csv', index=False, encoding='utf-8-sig')

print("\nData split and save: Completed.")


os.makedirs('/content/drive/MyDrive/MetS_Experiment/models', exist_ok=True)
os.makedirs('/content/drive/MyDrive/MetS_Experiment/results', exist_ok=True)


# [Step 1] Data load and scaling
print("1. Data load and scaling ...")
train_orig = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/preprocessed_data/train_original.csv')
train_smote = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/preprocessed_data/train_smote.csv')
val_df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/preprocessed_data/validation_set.csv')
test_df = pd.read_csv('/content/drive/MyDrive/MetS_Experiment/data/preprocessed_data/test_set.csv')

meta_cols = ['year', 'wt_itvex', 'wt_pool', 'kstrata', 'psu']

def get_X_y_w(df, has_weight=True):
    y = df['MS'].values
    w = df['wt_pool'].values if has_weight else np.ones(len(df))
    cols_to_drop = ['MS'] + [c for c in meta_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X, y, w

X_train, y_train, w_train = get_X_y_w(train_orig, has_weight=True)
X_train_smote, y_train_smote, w_train_smote = get_X_y_w(train_smote, has_weight=False)
X_val, y_val, w_val = get_X_y_w(val_df, has_weight=True)
X_test, y_test, w_test = get_X_y_w(test_df, has_weight=True)

continuous_cols = ['age', 'CO', 'NO2', 'O3', 'PM10', 'PM2_5', 'SO2', 'green_1']

# Scaler fitting and transformation
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_train_smote[continuous_cols] = scaler.transform(X_train_smote[continuous_cols])
X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

joblib.dump(scaler, '/content/drive/MyDrive/MetS_Experiment/models/standard_scaler.pkl')


# [Step 2] Deep learning model structure and evaluation function define
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

def train_dl_model(model_class, X_tr, y_tr, w_tr, pos_weight=None, patience=15, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = TensorDataset(torch.FloatTensor(X_tr.values), torch.FloatTensor(y_tr).unsqueeze(1), torch.FloatTensor(w_tr).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val).unsqueeze(1), torch.FloatTensor(w_val).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=g)  
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = model_class(X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if pos_weight:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter, best_weights = 0, None   

    for epoch in range(200):
        model.train()
        for b_X, b_y, b_w in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            (loss * b_w).mean().backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_X, b_y, b_w in val_loader:
                loss = criterion(model(b_X), b_y)
                val_loss += (loss * b_w).mean().item()

        if val_loss < best_val_loss:
            best_val_loss, patience_counter, best_weights = val_loss, 0, model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience: break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def get_dl_probs(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.FloatTensor(X.values))).numpy().flatten()

def calc_auc_ci(y_true, y_prob, w_true, n_bootstraps=1000):
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2: continue
        scores.append(roc_auc_score(y_true[indices], y_prob[indices], sample_weight=w_true[indices]))
    scores.sort()
    return scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores))]


# [Step 3] Model Training and Save
print("\n2. Model Training and Save: Start ...")
predictions = {}
imbalance_ratio = 2.54

ml_models = {
    'Logistic_Regression': LogisticRegression(C=0.0106, penalty='l2', random_state=42, class_weight='balanced'),
    'Random_Forest': RandomForestClassifier(n_estimators=380, max_depth=8, min_samples_split=6, max_features='sqrt', random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=580, max_depth=4, learning_rate=0.0101, subsample=0.923, colsample_bytree=0.755, random_state=42, eval_metric='logloss', scale_pos_weight=imbalance_ratio),
    'LightGBM': LGBMClassifier(n_estimators=250, num_leaves=20, learning_rate=0.0160, min_child_samples=10, subsample=0.802, colsample_bytree=0.845, random_state=42, class_weight='balanced'),
    'SVM': SVC(C=0.105, gamma='scale', probability=True, random_state=42, class_weight='balanced')
}

for name, model in ml_models.items():
    print(f"   - Training & Saving {name}...")
    model.fit(X_train, y_train, sample_weight=w_train)
    predictions[name] = model.predict_proba(X_test)[:, 1]
    joblib.dump(model, f'/content/drive/MyDrive/MetS_Experiment/models/{name}.pkl')

dl_models = {'BasicMLP': BasicMLP, 'DeepMLP': DeepMLP, 'ResNetMLP': ResNetMLP}
trained_dl_models = {}

for name, model_class in dl_models.items():
    print(f"   - Training & Saving {name}...")
    model = train_dl_model(model_class, X_train, y_train, w_train, pos_weight=imbalance_ratio)
    predictions[name] = get_dl_probs(model, X_test)
    trained_dl_models[name] = model
    torch.save(model.state_dict(), f'/content/drive/MyDrive/MetS_Experiment/models/{name}.pth')

results = []
for name, prob in predictions.items():
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, prob, sample_weight=w_test)
    auc_lower, auc_upper = calc_auc_ci(y_test, prob, w_test)
    auprc = average_precision_score(y_test, prob, sample_weight=w_test)
    acc = accuracy_score(y_test, pred, sample_weight=w_test)
    prec = precision_score(y_test, pred, sample_weight=w_test)
    rec = recall_score(y_test, pred, sample_weight=w_test)
    f1 = f1_score(y_test, pred, sample_weight=w_test)
    results.append([name, auc, f"[{auc_lower:.4f}-{auc_upper:.4f}]", auprc, f1, acc, prec, rec])

table8_df = pd.DataFrame(results, columns=['Model', 'AUC-ROC', 'AUC_95%_CI', 'AUPRC', 'F1-Score', 'Accuracy', 'Precision', 'Recall'])
table8_df = table8_df.sort_values(by='AUC-ROC', ascending=False)
table8_df.to_csv('/content/drive/MyDrive/MetS_Experiment/results/table8_weighted_performance.csv', index=False)
print(" Weighted performance Result is saved: '/MetS_Experiment/results/table8_weighted_performance.csv' .")

print("\nModel Training completed!")