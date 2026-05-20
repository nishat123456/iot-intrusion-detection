"""
CSC 428 - Machine Learning in Cybersecurity
Final Project — Step 1: Data Preprocessing & EDA
Author: M M Nishat

Dataset: CICIoT2023 (xxsmall sample)
Task:    Binary classification — DDoS-HTTP_Flood vs. BenignTraffic
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "CICIoT2023_xxsmall.csv")
FIGS   = os.path.join(BASE, "outputs", "figures")
RES    = os.path.join(BASE, "outputs", "results")
MODELS = os.path.join(BASE, "outputs", "models")
for d in [FIGS, RES, MODELS]:
    os.makedirs(d, exist_ok=True)

# ── 1. Load & filter to binary problem ────────────────────────────────────────
print("=" * 65)
print("  STEP 1 — DATA LOADING & PREPROCESSING")
print("=" * 65)

df_full = pd.read_csv(DATA)
print(f"\n[1] Full dataset loaded:  {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")
print(f"    Attack types: {df_full['label'].nunique()}")
print(f"\n    Label distribution (all classes):")
print(df_full['label'].value_counts().to_string())

# Binary subset: DDoS-HTTP_Flood vs. BenignTraffic (as per project proposal)
df = df_full[df_full['label'].isin(['DDoS-HTTP_Flood', 'BenignTraffic'])].copy()
df['label_enc'] = (df['label'] == 'DDoS-HTTP_Flood').astype(int)
print(f"\n[2] Binary subset:        {df.shape[0]:,} rows")
print(f"    BenignTraffic  (0): {(df['label_enc']==0).sum():,}")
print(f"    DDoS-HTTP_Flood(1): {(df['label_enc']==1).sum():,}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
# Drop index column and label text column
DROP_COLS = ['Unnamed: 0', 'label']
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

X = df.drop(columns=['label_enc'])
y = df['label_enc']

feature_names_original = X.columns.tolist()
print(f"\n[3] Features before selection: {len(feature_names_original)}")

# (a) Missing values
missing = X.isnull().sum()
print(f"\n[4] Missing values: {missing.sum()}")
if missing.sum() > 0:
    X.fillna(X.median(), inplace=True)
    print("    → Filled with column median")

# (b) Replace infinities
inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
print(f"    Infinite values: {inf_count}")
if inf_count > 0:
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    print("    → Replaced with column median")

# (c) Near-zero variance removal (variance < 1e-6)
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=1e-6)
vt.fit(X)
low_var = X.columns[~vt.get_support()].tolist()
if low_var:
    print(f"\n[5] Near-zero variance features removed ({len(low_var)}): {low_var}")
    X.drop(columns=low_var, inplace=True)
else:
    print(f"\n[5] No near-zero variance features found")

# (d) High-correlation removal (|r| > 0.98)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
if high_corr:
    print(f"[6] High-correlation features removed ({len(high_corr)}): {high_corr}")
    X.drop(columns=high_corr, inplace=True)
else:
    print(f"[6] No highly correlated features (threshold 0.98) found")

feature_names = X.columns.tolist()
print(f"\n[7] Final feature count: {len(feature_names)}")

# ── 3. EDA Plots ──────────────────────────────────────────────────────────────
# Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts()
ax.bar(['BenignTraffic\n(0)', 'DDoS-HTTP_Flood\n(1)'],
       [counts.get(0, 0), counts.get(1, 0)],
       color=['#2196F3', '#F44336'], edgecolor='black', linewidth=0.8)
ax.set_title('Class Distribution — Binary Subset', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Instances')
for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
    ax.text(i, v + 100, f'{v:,}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'class_distribution.png'), dpi=150)
plt.close()
print("\n[8] Saved: class_distribution.png")

# Feature correlation heatmap (top 20 features)
fig, ax = plt.subplots(figsize=(14, 11))
corr_sub = X[feature_names[:20]].corr()
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, annot=False, cmap='coolwarm',
            center=0, linewidths=0.3, ax=ax)
ax.set_title('Feature Correlation Heatmap (first 20 features)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("    Saved: correlation_heatmap.png")

# Feature distributions for top 6 features
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, feat in enumerate(feature_names[:6]):
    ax = axes[i // 3][i % 3]
    X_b = X[feat][y == 0]
    X_a = X[feat][y == 1]
    ax.hist(X_b, bins=50, alpha=0.6, color='#2196F3', label='Benign', density=True)
    ax.hist(X_a, bins=50, alpha=0.6, color='#F44336', label='DDoS', density=True)
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel('Density')
plt.suptitle('Feature Distributions: Benign vs DDoS-HTTP_Flood', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'feature_distributions.png'), dpi=150)
plt.close()
print("    Saved: feature_distributions.png")

# ── 4. Scaling ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

print(f"\n[9] StandardScaler applied (mean=0, std=1)")

# ── 5. Three train/test splits ────────────────────────────────────────────────
splits = {}
for ratio in [(0.60, 0.40), (0.70, 0.30), (0.80, 0.20)]:
    train_r, test_r = ratio
    tag = f"{int(train_r*100)}_{int(test_r*100)}"
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_r, random_state=42, stratify=y)
    splits[tag] = (X_tr, X_te, y_tr, y_te)
    print(f"    Split {int(train_r*100)}/{int(test_r*100)} — train: {len(X_tr):,}  test: {len(X_te):,}")

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(scaler,        os.path.join(MODELS, 'scaler.pkl'))
joblib.dump(splits,        os.path.join(RES,    'splits.pkl'))
joblib.dump(feature_names, os.path.join(RES,    'feature_names.pkl'))
X_scaled.to_csv(os.path.join(RES, 'X_scaled.csv'), index=False)
y.reset_index(drop=True).to_csv(os.path.join(RES, 'y.csv'), index=False)

# Save EDA summary
eda_summary = {
    'total_rows_full_dataset':   int(df_full.shape[0]),
    'total_attack_types':        int(df_full['label'].nunique()),
    'binary_subset_rows':        int(len(y)),
    'benign_count':              int((y == 0).sum()),
    'ddos_count':                int((y == 1).sum()),
    'original_features':         len(feature_names_original),
    'features_after_selection':  len(feature_names),
    'missing_values':            int(missing.sum()),
    'near_zero_var_removed':     len(low_var),
    'high_corr_removed':         len(high_corr),
}
pd.Series(eda_summary).to_csv(os.path.join(RES, 'eda_summary.csv'), header=['value'])

print(f"\n[10] Artifacts saved to outputs/")
print("\n" + "=" * 65)
print("  Preprocessing COMPLETE. Run 2_classical_models.py next.")
print("=" * 65)
