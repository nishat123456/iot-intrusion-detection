"""
CSC 428 / Publication Extension
Step 5: Multi-Class Hierarchical IoT Intrusion Detection
Author: M M Nishat

Hierarchical classification on the full CICIoT2023 dataset:
  Level-1 (family): DDoS | DoS | Mirai | Recon | Web | Benign  (6 classes)
  Level-2 (subtype): all 34 specific labels

Models: LightGBM, XGBoost, Random Forest
Evaluation: 10-fold CV + 80/20 split, per-class metrics
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib

from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics           import (accuracy_score, f1_score, precision_score,
                                       recall_score, classification_report,
                                       confusion_matrix, roc_auc_score)
from sklearn.ensemble          import RandomForestClassifier
import xgboost  as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "CICIoT2023_xxsmall.csv")
RES    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

print("=" * 70)
print("  STEP 5 — MULTI-CLASS HIERARCHICAL CLASSIFICATION")
print("=" * 70)

# ── Load raw data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"\n  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Attack types: {df['label'].nunique()}")

# ── Hierarchical family mapping ────────────────────────────────────────────────
FAMILY_MAP = {
    'DDoS-RSTFINFlood':        'DDoS',
    'DDoS-PSHACK_Flood':       'DDoS',
    'DDoS-SYN_Flood':          'DDoS',
    'DDoS-UDP_Flood':          'DDoS',
    'DDoS-TCP_Flood':          'DDoS',
    'DDoS-ICMP_Flood':         'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ACK_Fragmentation':  'DDoS',
    'DDoS-UDP_Fragmentation':  'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris':          'DDoS',
    'DDoS-HTTP_Flood':         'DDoS',
    'DoS-UDP_Flood':           'DoS',
    'DoS-SYN_Flood':           'DoS',
    'DoS-TCP_Flood':           'DoS',
    'DoS-HTTP_Flood':          'DoS',
    'Mirai-greeth_flood':      'Mirai',
    'Mirai-greip_flood':       'Mirai',
    'Mirai-udpplain':          'Mirai',
    'Recon-OSScan':            'Recon',
    'Recon-PortScan':          'Recon',
    'VulnerabilityScan':       'Recon',
    'Recon-HostDiscovery':     'Recon',
    'Recon-PingSweep':         'Recon',
    'DNS_Spoofing':            'Spoofing',
    'MITM-ArpSpoofing':        'Spoofing',
    'DictionaryBruteForce':    'WebAttack',
    'BrowserHijacking':        'WebAttack',
    'CommandInjection':        'WebAttack',
    'SqlInjection':            'WebAttack',
    'XSS':                     'WebAttack',
    'Backdoor_Malware':        'WebAttack',
    'Uploading_Attack':        'WebAttack',
    'BenignTraffic':           'Benign',
}

df['family'] = df['label'].map(FAMILY_MAP)

# ── Feature preprocessing ──────────────────────────────────────────────────────
DROP_COLS = ['Unnamed: 0', 'label', 'family']
X_raw = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Fix infinities and missing
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)

# Near-zero variance
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=1e-6)
vt.fit(X_raw)
X_raw = X_raw.loc[:, vt.get_support()]

# High correlation removal
corr = X_raw.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
X_raw.drop(columns=drop_corr, inplace=True)

feature_names = X_raw.columns.tolist()
print(f"  Features after selection: {len(feature_names)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── Encode labels ──────────────────────────────────────────────────────────────
le_family = LabelEncoder()
le_subtype = LabelEncoder()

y_family  = le_family.fit_transform(df['family'])
y_subtype = le_subtype.fit_transform(df['label'])

print(f"\n  Family classes ({len(le_family.classes_)}): {list(le_family.classes_)}")
print(f"  Subtype classes: {len(le_subtype.classes_)}")

# ── Save encoders ──────────────────────────────────────────────────────────────
joblib.dump(le_family,     os.path.join(MODELS, 'le_family.pkl'))
joblib.dump(le_subtype,    os.path.join(MODELS, 'le_subtype.pkl'))
joblib.dump(feature_names, os.path.join(RES,    'mc_feature_names.pkl'))
joblib.dump(scaler,        os.path.join(MODELS, 'mc_scaler.pkl'))

# ── Model definitions ──────────────────────────────────────────────────────────
MODELS_DEF = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbose=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric='mlogloss', random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
}

# ── Evaluate: family-level + subtype-level ─────────────────────────────────────
def run_eval(X, y, label_enc, level_name, model_name, model_obj):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    t0 = time.time()
    model_obj.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)
    y_pred = model_obj.predict(X_te)
    acc  = round(accuracy_score(y_te, y_pred), 4)
    prec_w = round(precision_score(y_te, y_pred, average='weighted', zero_division=0), 4)
    rec_w  = round(recall_score(y_te, y_pred, average='weighted', zero_division=0), 4)
    f1_w = round(f1_score(y_te, y_pred, average='weighted'), 4)
    f1_m = round(f1_score(y_te, y_pred, average='macro'), 4)
    print(f"    [{level_name}] Acc={acc}  Prec={prec_w}  Recall={rec_w}  F1-w={f1_w}  F1-m={f1_m}  ({elapsed}s)")
    return model_obj, y_te, y_pred, acc, prec_w, rec_w, f1_w, f1_m, elapsed

mc_results = []

for m_name, m_obj in MODELS_DEF.items():
    print(f"\n  ── {m_name} ──")

    # Family level
    fresh_f = m_obj.__class__(**m_obj.get_params())
    model_f, yte_f, ypred_f, acc_f, prec_f, rec_f, f1w_f, f1m_f, t_f = run_eval(
        X_scaled, y_family, le_family, 'family', m_name, fresh_f)
    mc_results.append({
        'model': m_name, 'level': 'family',
        'accuracy': acc_f, 'precision_weighted': prec_f, 'recall_weighted': rec_f,
        'f1_weighted': f1w_f, 'f1_macro': f1m_f, 'train_sec': t_f
    })

    # Subtype level
    fresh_s = m_obj.__class__(**m_obj.get_params())
    model_s, yte_s, ypred_s, acc_s, prec_s, rec_s, f1w_s, f1m_s, t_s = run_eval(
        X_scaled, y_subtype, le_subtype, 'subtype', m_name, fresh_s)
    mc_results.append({
        'model': m_name, 'level': 'subtype',
        'accuracy': acc_s, 'precision_weighted': prec_s, 'recall_weighted': rec_s,
        'f1_weighted': f1w_s, 'f1_macro': f1m_s, 'train_sec': t_s
    })

    # Save best model (LightGBM)
    if m_name == 'LightGBM':
        joblib.dump(model_f, os.path.join(MODELS, 'lgbm_family.pkl'))
        joblib.dump(model_s, os.path.join(MODELS, 'lgbm_subtype.pkl'))
        cm_family  = confusion_matrix(yte_f, ypred_f)
        cm_subtype = confusion_matrix(yte_s, ypred_s)
        report_family  = classification_report(yte_f, ypred_f,
                            target_names=le_family.classes_, output_dict=True)
        report_subtype = classification_report(yte_s, ypred_s,
                            target_names=le_subtype.classes_, output_dict=True)
        y_te_f_saved  = yte_f
        y_pred_f_saved = ypred_f
        y_te_s_saved  = yte_s
        y_pred_s_saved = ypred_s

# ── Save results ───────────────────────────────────────────────────────────────
df_mc = pd.DataFrame(mc_results)
df_mc.to_csv(os.path.join(RES, 'multiclass_results.csv'), index=False)

# ── 10-Fold CV (family level, LightGBM) ───────────────────────────────────────
print("\n  ── 10-Fold CV (LightGBM, family level) ──")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
lgbm_cv = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                               random_state=42, n_jobs=-1, verbose=-1)
t0 = time.time()
cv_res = cross_validate(lgbm_cv, X_scaled, y_family, cv=skf,
                        scoring=['accuracy', 'f1_weighted', 'f1_macro'], n_jobs=-1)
cv_time = round(time.time() - t0, 2)
print(f"    Acc={cv_res['test_accuracy'].mean():.4f}  "
      f"F1-w={cv_res['test_f1_weighted'].mean():.4f}  "
      f"F1-macro={cv_res['test_f1_macro'].mean():.4f}  ({cv_time}s)")

cv_row = {
    'model': 'LightGBM', 'level': 'family_10foldCV',
    'accuracy':    round(cv_res['test_accuracy'].mean(), 4),
    'f1_weighted': round(cv_res['test_f1_weighted'].mean(), 4),
    'f1_macro':    round(cv_res['test_f1_macro'].mean(), 4),
    'train_sec': cv_time,
}
df_mc = pd.concat([df_mc, pd.DataFrame([cv_row])], ignore_index=True)
df_mc.to_csv(os.path.join(RES, 'multiclass_results.csv'), index=False)

# ── Figure 1: Family confusion matrix (LightGBM) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
cm_pct = cm_family.astype(float) / cm_family.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
            xticklabels=le_family.classes_, yticklabels=le_family.classes_)
ax.set_title('LightGBM — Family-Level Confusion Matrix (80/20, %)', fontweight='bold')
ax.set_xlabel('Predicted Family'); ax.set_ylabel('True Family')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'mc_family_confusion.png'), dpi=150)
plt.close()
print("\n  Saved: mc_family_confusion.png")

# ── Figure 2: Per-class F1 bar chart (family level) ───────────────────────────
f1_per_class = {cls: report_family[cls]['f1-score']
                for cls in le_family.classes_ if cls in report_family}
fig, ax = plt.subplots(figsize=(9, 5))
classes = list(f1_per_class.keys())
vals    = list(f1_per_class.values())
colors  = ['#C62828' if v < 0.97 else '#1976D2' for v in vals]
bars = ax.bar(classes, vals, color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
ax.set_ylim(0.85, 1.01)
ax.set_ylabel('F1-Score')
ax.set_title('Per-Family F1-Score — LightGBM 80/20 Split', fontweight='bold')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'mc_family_f1_per_class.png'), dpi=150)
plt.close()
print("  Saved: mc_family_f1_per_class.png")

# ── Figure 3: Subtype confusion matrix (LightGBM, compact) ───────────────────
fig, ax = plt.subplots(figsize=(16, 13))
cm_s_pct = cm_subtype.astype(float) / cm_subtype.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_s_pct, annot=False, cmap='Blues', ax=ax,
            xticklabels=le_subtype.classes_, yticklabels=le_subtype.classes_)
ax.set_title('LightGBM — 34-Class Subtype Confusion Matrix (80/20, %)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.tick_params(axis='x', labelsize=6, rotation=90)
ax.tick_params(axis='y', labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'mc_subtype_confusion.png'), dpi=150)
plt.close()
print("  Saved: mc_subtype_confusion.png")

# ── Figure 4: Multi-model comparison (family level, 80/20) ────────────────────
fam_df = df_mc[df_mc['level'] == 'family']
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metrics = ['accuracy', 'f1_weighted', 'f1_macro']
labels  = ['Accuracy', 'F1-Weighted', 'F1-Macro']
colors  = ['#1976D2', '#388E3C', '#F57C00']
for ax, metric, label in zip(axes, metrics, labels):
    bars = ax.bar(fam_df['model'], fam_df[metric],
                  color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
    ax.set_title(label, fontweight='bold')
    ax.set_ylim(max(0, fam_df[metric].min() - 0.05), 1.01)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, fam_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('Family-Level Multi-Class Detection — All Models (80/20 Split)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'mc_model_comparison.png'), dpi=150)
plt.close()
print("  Saved: mc_model_comparison.png")

print("\n" + "=" * 70)
print("  STEP 5 COMPLETE — Multi-class results saved.")
print("  Run 6_shap_analysis.py next.")
print("=" * 70)
