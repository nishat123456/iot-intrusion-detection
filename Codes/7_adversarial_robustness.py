"""
CSC 428 / Publication Extension
Step 7: Adversarial Robustness Analysis
Author: M M Nishat

Evaluates model robustness against feature-space perturbations:
  1. Gaussian noise at σ = 0.1, 0.25, 0.5, 1.0, 2.0 (all features)
  2. Targeted perturbation — noise on top-10 SHAP features only
  3. Zero-out ablation — zero top-k SHAP features, measure accuracy drop

Uses the LightGBM family-level classifier trained in Step 5.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

print("=" * 70)
print("  STEP 7 — ADVERSARIAL ROBUSTNESS ANALYSIS")
print("=" * 70)

# ── Load model + data ──────────────────────────────────────────────────────────
model         = joblib.load(os.path.join(MODELS, 'lgbm_family.pkl'))
scaler        = joblib.load(os.path.join(MODELS, 'mc_scaler.pkl'))
le_family     = joblib.load(os.path.join(MODELS, 'le_family.pkl'))
feature_names = joblib.load(os.path.join(RES,    'mc_feature_names.pkl'))

DATA = os.path.join(BASE, "data", "CICIoT2023_xxsmall.csv")
df_raw = pd.read_csv(DATA)

FAMILY_MAP = {
    'DDoS-RSTFINFlood':'DDoS','DDoS-PSHACK_Flood':'DDoS','DDoS-SYN_Flood':'DDoS',
    'DDoS-UDP_Flood':'DDoS','DDoS-TCP_Flood':'DDoS','DDoS-ICMP_Flood':'DDoS',
    'DDoS-SynonymousIP_Flood':'DDoS','DDoS-ACK_Fragmentation':'DDoS',
    'DDoS-UDP_Fragmentation':'DDoS','DDoS-ICMP_Fragmentation':'DDoS',
    'DDoS-SlowLoris':'DDoS','DDoS-HTTP_Flood':'DDoS',
    'DoS-UDP_Flood':'DoS','DoS-SYN_Flood':'DoS','DoS-TCP_Flood':'DoS','DoS-HTTP_Flood':'DoS',
    'Mirai-greeth_flood':'Mirai','Mirai-greip_flood':'Mirai','Mirai-udpplain':'Mirai',
    'Recon-OSScan':'Recon','Recon-PortScan':'Recon','VulnerabilityScan':'Recon',
    'Recon-HostDiscovery':'Recon','Recon-PingSweep':'Recon',
    'DNS_Spoofing':'Spoofing','MITM-ArpSpoofing':'Spoofing',
    'DictionaryBruteForce':'WebAttack','BrowserHijacking':'WebAttack',
    'CommandInjection':'WebAttack','SqlInjection':'WebAttack',
    'XSS':'WebAttack','Backdoor_Malware':'WebAttack','Uploading_Attack':'WebAttack',
    'BenignTraffic':'Benign',
}
df_raw['family'] = df_raw['label'].map(FAMILY_MAP)
X_raw = df_raw.drop(columns=[c for c in ['Unnamed: 0','label','family'] if c in df_raw.columns])
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)
vt = VarianceThreshold(threshold=1e-6); vt.fit(X_raw)
X_raw = X_raw.loc[:, vt.get_support()]
corr = X_raw.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
X_raw.drop(columns=drop_corr, inplace=True)
X_scaled_full = scaler.transform(X_raw)
y_family = le_family.transform(df_raw['family'])

# 20% holdout (same split as Step 5)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled_full, y_family, test_size=0.20, random_state=42, stratify=y_family)

# Baseline
y_baseline = model.predict(X_te)
baseline_acc = accuracy_score(y_te, y_baseline)
baseline_f1  = f1_score(y_te, y_baseline, average='weighted')
print(f"\n  Baseline — Acc={baseline_acc:.4f}  F1-w={baseline_f1:.4f}")

# ── Load SHAP importance ranking ───────────────────────────────────────────────
shap_table_path = os.path.join(RES, 'shap_importance_table.csv')
if os.path.exists(shap_table_path):
    shap_table = pd.read_csv(shap_table_path, index_col=0)
    top_shap_features = shap_table['global_mean'].sort_values(ascending=False).head(10).index.tolist()
    top_shap_idx = [feature_names.index(f) for f in top_shap_features if f in feature_names]
else:
    # Fallback: use first 10 features
    top_shap_idx = list(range(10))
    top_shap_features = [feature_names[i] for i in top_shap_idx]

print(f"  Top-10 SHAP features for targeted perturbation: {top_shap_features[:5]}…")

# ── Experiment 1: Global Gaussian noise ───────────────────────────────────────
SIGMAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
global_rows = []
print("\n  [1] Global Gaussian noise:")
for sigma in SIGMAS:
    rng = np.random.RandomState(0)
    X_noisy = X_te + rng.normal(0, sigma, X_te.shape)
    y_pred  = model.predict(X_noisy)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='weighted')
    drop_acc = round((baseline_acc - acc) * 100, 3)
    print(f"    σ={sigma:.2f}  Acc={acc:.4f}  F1={f1:.4f}  ΔAcc=-{drop_acc}%")
    global_rows.append({'sigma': sigma, 'scope': 'global',
                        'accuracy': acc, 'f1_weighted': f1, 'acc_drop': drop_acc})

# ── Experiment 2: Targeted noise (top SHAP features only) ────────────────────
targeted_rows = []
print("\n  [2] Targeted noise (top-10 SHAP features only):")
for sigma in SIGMAS:
    rng = np.random.RandomState(0)
    X_noisy = X_te.copy()
    X_noisy[:, top_shap_idx] += rng.normal(0, sigma, (X_te.shape[0], len(top_shap_idx)))
    y_pred = model.predict(X_noisy)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='weighted')
    drop_acc = round((baseline_acc - acc) * 100, 3)
    print(f"    σ={sigma:.2f}  Acc={acc:.4f}  F1={f1:.4f}  ΔAcc=-{drop_acc}%")
    targeted_rows.append({'sigma': sigma, 'scope': 'targeted_top10_shap',
                          'accuracy': acc, 'f1_weighted': f1, 'acc_drop': drop_acc})

# ── Experiment 3: Feature ablation (zero-out top-k features) ──────────────────
ablation_rows = []
K_VALUES = [1, 3, 5, 10, 15, 20]
print("\n  [3] Feature ablation (zero-out top-k SHAP features):")
shap_sorted_idx = [feature_names.index(f) for f in
                   shap_table['global_mean'].sort_values(ascending=False).index
                   if f in feature_names] if os.path.exists(shap_table_path) else list(range(30))

for k in K_VALUES:
    X_ablated = X_te.copy()
    X_ablated[:, shap_sorted_idx[:k]] = 0.0
    y_pred = model.predict(X_ablated)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='weighted')
    drop_acc = round((baseline_acc - acc) * 100, 3)
    print(f"    k={k:2d}  Acc={acc:.4f}  F1={f1:.4f}  ΔAcc=-{drop_acc}%")
    ablation_rows.append({'k': k, 'accuracy': acc, 'f1_weighted': f1, 'acc_drop': drop_acc})

# ── Save tables ────────────────────────────────────────────────────────────────
df_robustness = pd.DataFrame(global_rows + targeted_rows)
df_robustness.to_csv(os.path.join(RES, 'robustness_noise.csv'), index=False)
df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(os.path.join(RES, 'robustness_ablation.csv'), index=False)

# ── Figure 1: Accuracy vs. sigma (global vs. targeted) ────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
df_g = pd.DataFrame(global_rows)
df_t = pd.DataFrame(targeted_rows)
ax.plot(df_g['sigma'], df_g['accuracy'], marker='o', color='#1976D2',
        linewidth=2, label='Global noise (all features)', markersize=7)
ax.plot(df_t['sigma'], df_t['accuracy'], marker='s', color='#C62828',
        linewidth=2, linestyle='--', label='Targeted noise (top-10 SHAP features)', markersize=7)
ax.axhline(baseline_acc, color='gray', linestyle=':', linewidth=1.5, label=f'Baseline ({baseline_acc:.4f})')
ax.set_xlabel('Gaussian Noise σ (in feature space)', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Model Robustness — Accuracy vs. Perturbation Magnitude',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(max(0, min(df_g['accuracy'].min(), df_t['accuracy'].min()) - 0.05), 1.005)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'robustness_noise_curve.png'), dpi=150)
plt.close()
print("\n  Saved: robustness_noise_curve.png")

# ── Figure 2: Feature ablation curve ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df_ablation['k'], df_ablation['accuracy'], marker='D',
        color='#7B1FA2', linewidth=2, markersize=7)
ax.axhline(baseline_acc, color='gray', linestyle=':', linewidth=1.5, label=f'Baseline ({baseline_acc:.4f})')
for _, row in df_ablation.iterrows():
    ax.text(row['k'], row['accuracy'] + 0.002,
            f"{row['accuracy']:.3f}", ha='center', fontsize=8)
ax.set_xlabel('Number of Top SHAP Features Zeroed Out', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Feature Ablation — Accuracy Drop as Top Features are Removed',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(max(0, df_ablation['accuracy'].min() - 0.1), 1.005)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'robustness_ablation_curve.png'), dpi=150)
plt.close()
print("  Saved: robustness_ablation_curve.png")

# ── Figure 3: Accuracy drop heatmap (per-family, σ=0.5 global noise) ─────────
print("\n  [4] Per-family accuracy drop at σ=0.5:")
sigma_test = 0.5
rng = np.random.RandomState(0)
X_noisy = X_te + rng.normal(0, sigma_test, X_te.shape)
y_pred_noisy = model.predict(X_noisy)
y_pred_clean = model.predict(X_te)

family_drop = {}
for fi, fname in enumerate(le_family.classes_):
    mask = y_te == fi
    if mask.sum() == 0:
        continue
    acc_clean = accuracy_score(y_te[mask], y_pred_clean[mask])
    acc_noisy = accuracy_score(y_te[mask], y_pred_noisy[mask])
    drop = round((acc_clean - acc_noisy) * 100, 2)
    family_drop[fname] = {'clean': round(acc_clean, 4), 'noisy': round(acc_noisy, 4), 'drop_%': drop}
    print(f"    {fname:12s}  clean={acc_clean:.4f}  noisy={acc_noisy:.4f}  drop={drop:.2f}%")

df_family_drop = pd.DataFrame(family_drop).T
df_family_drop.to_csv(os.path.join(RES, 'robustness_per_family_drop.csv'))

fig, ax = plt.subplots(figsize=(9, 5))
classes_sorted = df_family_drop.sort_values('drop_%', ascending=False)
colors = ['#C62828' if v > 5 else '#1976D2' for v in classes_sorted['drop_%']]
bars = ax.bar(classes_sorted.index, classes_sorted['drop_%'],
              color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
for bar, v in zip(bars, classes_sorted['drop_%']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Accuracy Drop (%)', fontsize=11)
ax.set_xlabel('Attack Family', fontsize=11)
ax.set_title(f'Per-Family Accuracy Drop under Gaussian Noise (σ={sigma_test})',
             fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'robustness_per_family_drop.png'), dpi=150)
plt.close()
print("  Saved: robustness_per_family_drop.png")

print("\n" + "=" * 70)
print("  STEP 7 COMPLETE — Adversarial robustness analysis done.")
print("  Run 8_publication_summary.py next.")
print("=" * 70)
