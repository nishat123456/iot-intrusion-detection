"""
CSC 428 - Machine Learning in Cybersecurity
Final Project — Step 2: Classical Models + 10-Fold CV
Author: M M Nishat

Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Splits: 60/40, 70/30, 80/20  +  10-Fold Cross-Validation
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     roc_curve)
import xgboost  as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

# ── Load preprocessed data ────────────────────────────────────────────────────
splits        = joblib.load(os.path.join(RES, 'splits.pkl'))
feature_names = joblib.load(os.path.join(RES, 'feature_names.pkl'))
X_all = pd.read_csv(os.path.join(RES, 'X_scaled.csv'))
y_all = pd.read_csv(os.path.join(RES, 'y.csv')).squeeze()

print("=" * 65)
print("  STEP 2 — CLASSICAL MODELS TRAINING & EVALUATION")
print("=" * 65)
print(f"  Total samples: {len(y_all):,}  |  Features: {len(feature_names)}")

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS_DEF = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbose=-1),
}

# ── Helper: evaluate one split ────────────────────────────────────────────────
def evaluate_split(model, X_train, X_test, y_train, y_test, model_name, split_tag):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'model':     model_name,
        'split':     split_tag,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'train_sec': round(train_time, 3),
    }, y_pred, y_prob

# ── Helper: 10-fold CV ────────────────────────────────────────────────────────
def cv_evaluate(model, X, y, model_name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    t0 = time.time()
    cv_res = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    cv_time = time.time() - t0
    return {
        'model':     model_name,
        'split':     '10-fold CV',
        'accuracy':  cv_res['test_accuracy'].mean(),
        'precision': cv_res['test_precision'].mean(),
        'recall':    cv_res['test_recall'].mean(),
        'f1':        cv_res['test_f1'].mean(),
        'roc_auc':   cv_res['test_roc_auc'].mean(),
        'train_sec': round(cv_time, 3),
    }

# ── Run all evaluations ───────────────────────────────────────────────────────
all_results  = []
cm_data      = {}   # confusion matrices
roc_data     = {}   # ROC curves (for 80/20 split)
fi_data      = {}   # feature importances

SPLIT_LABELS = {'60_40': '60/40', '70_30': '70/30', '80_20': '80/20'}

for m_name, m_obj in MODELS_DEF.items():
    print(f"\n  ── {m_name} ──")
    cm_data[m_name]  = {}
    roc_data[m_name] = {}

    for tag, (X_tr, X_te, y_tr, y_te) in splits.items():
        res, y_pred, y_prob = evaluate_split(
            m_obj.__class__(**m_obj.get_params()),   # fresh instance
            X_tr, X_te, y_tr, y_te, m_name, SPLIT_LABELS[tag])
        all_results.append(res)
        cm_data[m_name][SPLIT_LABELS[tag]] = confusion_matrix(y_te, y_pred)
        if tag == '80_20':
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_data[m_name] = (fpr, tpr, res['roc_auc'])
        print(f"    {SPLIT_LABELS[tag]}  Acc={res['accuracy']:.4f}  "
              f"F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}  "
              f"({res['train_sec']}s)")

    # 10-Fold CV
    cv_res = cv_evaluate(m_obj.__class__(**m_obj.get_params()), X_all, y_all, m_name)
    all_results.append(cv_res)
    print(f"    10-fold  Acc={cv_res['accuracy']:.4f}  "
          f"F1={cv_res['f1']:.4f}  AUC={cv_res['roc_auc']:.4f}")

    # Feature importance (save for tree models)
    if m_name in ('Random Forest', 'LightGBM', 'XGBoost'):
        _, X_te, _, _ = splits['80_20']
        X_tr, _, y_tr, _ = splits['80_20']
        fresh = m_obj.__class__(**m_obj.get_params())
        fresh.fit(splits['80_20'][0], splits['80_20'][2])
        fi_data[m_name] = dict(zip(feature_names, fresh.feature_importances_))

    # Save model (80/20 trained)
    fresh = m_obj.__class__(**m_obj.get_params())
    fresh.fit(splits['80_20'][0], splits['80_20'][2])
    safe_name = m_name.lower().replace(' ', '_')
    joblib.dump(fresh, os.path.join(MODELS, f'{safe_name}.pkl'))

# ── Save results table ─────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_results)
df_res = df_res.round(4)
df_res.to_csv(os.path.join(RES, 'classical_results.csv'), index=False)
print(f"\n\n{'='*65}")
print("  RESULTS TABLE")
print(f"{'='*65}")
print(df_res.to_string(index=False))

# ── Figure 1: Metric comparison bar chart ────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
split_order = ['60/40', '70/30', '80/20', '10-fold CV']
model_order  = list(MODELS_DEF.keys())
x = np.arange(len(model_order))
w = 0.20

for ax, metric, label in zip(axes, metrics, metric_labels):
    for i, split in enumerate(split_order):
        vals = [df_res[(df_res['model']==m) & (df_res['split']==split)][metric].values
                for m in model_order]
        vals = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + i*w - 0.30, vals, w, label=split, color=colors[i], alpha=0.85)
    ax.set_title(label, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_order], fontsize=8)
    ax.set_ylim(0.90, 1.01)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Model Performance Comparison Across Train/Test Splits & 10-Fold CV',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'metric_comparison.png'), dpi=150)
plt.close()
print("\n  Saved: metric_comparison.png")

# ── Figure 2: Confusion matrices (80/20 split) ───────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, m_name in zip(axes, model_order):
    cm = cm_data[m_name]['80/20']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'DDoS'], yticklabels=['Benign', 'DDoS'])
    ax.set_title(m_name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices — 80/20 Split', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'confusion_matrices.png'), dpi=150)
plt.close()
print("  Saved: confusion_matrices.png")

# ── Figure 3: ROC curves (80/20 split) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
roc_colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
for (m_name, (fpr, tpr, auc)), color in zip(roc_data.items(), roc_colors):
    ax.plot(fpr, tpr, lw=2, color=color, label=f'{m_name} (AUC={auc:.4f})')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — 80/20 Split', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'roc_curves.png'), dpi=150)
plt.close()
print("  Saved: roc_curves.png")

# ── Figure 4: Feature importance (LightGBM) ──────────────────────────────────
if 'LightGBM' in fi_data:
    fi_ser = pd.Series(fi_data['LightGBM']).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi_ser[::-1].plot(kind='barh', ax=ax, color='#1976D2', edgecolor='black', linewidth=0.5)
    ax.set_title('Top 15 Feature Importances — LightGBM (80/20 split)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'feature_importance_lgbm.png'), dpi=150)
    plt.close()
    print("  Saved: feature_importance_lgbm.png")

if 'Random Forest' in fi_data:
    fi_ser = pd.Series(fi_data['Random Forest']).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi_ser[::-1].plot(kind='barh', ax=ax, color='#388E3C', edgecolor='black', linewidth=0.5)
    ax.set_title('Top 15 Feature Importances — Random Forest (80/20 split)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'feature_importance_rf.png'), dpi=150)
    plt.close()
    print("  Saved: feature_importance_rf.png")

print("\n" + "=" * 65)
print("  Classical models COMPLETE. Run 3_lstm_model.py next.")
print("=" * 65)
