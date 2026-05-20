"""
CSC 428 / Publication Extension
Step 6: SHAP Explainability Analysis
Author: M M Nishat

SHAP TreeExplainer on LightGBM (family-level classifier).
Generates: summary plot, per-class mean |SHAP|, top-feature heatmap.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

print("=" * 70)
print("  STEP 6 — SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

# ── Load artifacts from Step 5 ────────────────────────────────────────────────
model        = joblib.load(os.path.join(MODELS, 'lgbm_family.pkl'))
scaler       = joblib.load(os.path.join(MODELS, 'mc_scaler.pkl'))
le_family    = joblib.load(os.path.join(MODELS, 'le_family.pkl'))
feature_names = joblib.load(os.path.join(RES,   'mc_feature_names.pkl'))

import pandas as _pd, numpy as _np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

DATA = os.path.join(BASE, "data", "CICIoT2023_xxsmall.csv")
df_raw = _pd.read_csv(DATA)

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
X_raw.replace([_np.inf, -_np.inf], _np.nan, inplace=True)
X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)
vt = VarianceThreshold(threshold=1e-6); vt.fit(X_raw)
X_raw = X_raw.loc[:, vt.get_support()]
corr = X_raw.corr().abs()
upper = corr.where(_np.triu(_np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
X_raw.drop(columns=drop_corr, inplace=True)
X_scaled_full = scaler.transform(X_raw)

# Stratified sample for SHAP (5,000 rows — keeps it fast)
from sklearn.model_selection import train_test_split
y_family = le_family.transform(df_raw['family'])
_, X_shap, _, y_shap = train_test_split(
    X_scaled_full, y_family, test_size=5000/len(y_family),
    random_state=42, stratify=y_family)
X_shap_df = pd.DataFrame(X_shap, columns=feature_names)

print(f"\n  SHAP sample size: {len(X_shap_df)} rows")
print(f"  Computing TreeExplainer SHAP values…")

explainer   = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_shap_df)

print("  Done.")

# Normalize: SHAP >= 0.41 returns (n_samples, n_features, n_classes); older returns list
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    # (n_samples, n_features, n_classes) → list of (n_samples, n_features)
    shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
else:
    shap_values = shap_values_raw

# ── Figure 1: SHAP beeswarm / summary (mean |SHAP| across all classes) ────────
shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (n_samples, n_features)

fig, ax = plt.subplots(figsize=(10, 8))
mean_abs = pd.Series(shap_abs_mean.mean(axis=0), index=feature_names).sort_values(ascending=False)
top20 = mean_abs.head(20)
top20[::-1].plot(kind='barh', ax=ax, color='#1976D2', edgecolor='black', linewidth=0.5)
ax.set_title('Top 20 Features — Mean |SHAP Value| (All Attack Families)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Mean |SHAP Value|')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'shap_global_importance.png'), dpi=150)
plt.close()
print("\n  Saved: shap_global_importance.png")

# ── Figure 2: Per-class top-10 SHAP heatmap ───────────────────────────────────
class_names = le_family.classes_
top_features = mean_abs.head(15).index.tolist()
heatmap_data = pd.DataFrame(index=class_names, columns=top_features, dtype=float)
for i, cls in enumerate(class_names):
    sv = np.abs(shap_values[i])
    heatmap_data.loc[cls] = pd.Series(sv.mean(axis=0), index=feature_names)[top_features].values

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_data.astype(float), cmap='YlOrRd', ax=ax,
            linewidths=0.3, annot=True, fmt='.3f', annot_kws={'size': 7})
ax.set_title('Per-Family Mean |SHAP Value| — Top 15 Features',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Feature'); ax.set_ylabel('Attack Family')
ax.tick_params(axis='x', rotation=45, labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'shap_per_family_heatmap.png'), dpi=150)
plt.close()
print("  Saved: shap_per_family_heatmap.png")

# ── Figure 3: SHAP summary dot plot for 3 most interesting classes ─────────────
interesting = ['DDoS', 'WebAttack', 'Recon']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, cls in zip(axes, interesting):
    ci = list(class_names).index(cls)
    sv_cls = shap_values[ci]
    top_idx = np.abs(sv_cls).mean(axis=0).argsort()[::-1][:10]
    top_feat = [feature_names[i] for i in top_idx]
    top_vals = sv_cls[:, top_idx]
    top_data = X_shap_df[top_feat].values
    for j, (feat, vals, data) in enumerate(
            zip(top_feat[::-1], top_vals.T[::-1], top_data.T[::-1])):
        y_jitter = np.full(len(vals), j) + np.random.normal(0, 0.05, len(vals))
        sc = ax.scatter(vals[:500], y_jitter[:500], c=data[:500],
                        cmap='coolwarm', alpha=0.4, s=4, vmin=-2, vmax=2)
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat[::-1], fontsize=8)
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_title(f'SHAP — {cls}', fontweight='bold')
    ax.set_xlabel('SHAP value')
    ax.grid(axis='x', alpha=0.2)
plt.suptitle('SHAP Dot Plot — Top 10 Features per Attack Family',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'shap_dot_plots.png'), dpi=150)
plt.close()
print("  Saved: shap_dot_plots.png")

# ── Save SHAP mean importances table ──────────────────────────────────────────
shap_table = pd.DataFrame(
    {cls: np.abs(shap_values[i]).mean(axis=0) for i, cls in enumerate(class_names)},
    index=feature_names
)
shap_table['global_mean'] = shap_table.mean(axis=1)
shap_table = shap_table.sort_values('global_mean', ascending=False)
shap_table.to_csv(os.path.join(RES, 'shap_importance_table.csv'))
print("  Saved: shap_importance_table.csv")

print("\n" + "=" * 70)
print("  STEP 6 COMPLETE — SHAP analysis done.")
print("  Run 7_adversarial_robustness.py next.")
print("=" * 70)
