"""
CSC 428 - Machine Learning in Cybersecurity
Final Project — Step 4: Final Summary Table & Combined Figures
Author: M M Nishat

Merges classical + LSTM results into one master table and
generates the comparison figures needed for the report.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, "outputs", "results")
FIGS  = os.path.join(BASE, "outputs", "figures")

# ── Load results ───────────────────────────────────────────────────────────────
classical = pd.read_csv(os.path.join(RES, 'classical_results.csv'))
lstm      = pd.read_csv(os.path.join(RES, 'lstm_results.csv'))
# Add missing epoch column to classical
if 'epochs' not in classical.columns:
    classical['epochs'] = '-'
df = pd.concat([classical, lstm], ignore_index=True)

print("=" * 65)
print("  STEP 4 — FINAL SUMMARY")
print("=" * 65)

# ── Master results table ───────────────────────────────────────────────────────
print("\n  FULL RESULTS TABLE")
print(df[['model','split','accuracy','precision','recall','f1','roc_auc','train_sec']
         ].to_string(index=False))

# Best config per model (10-fold CV)
cv_only = df[df['split'] == '10-fold CV'][
    ['model','accuracy','precision','recall','f1','roc_auc']].set_index('model')
print("\n  10-FOLD CV SUMMARY (mean across folds)")
print(cv_only.round(4).to_string())

df.to_csv(os.path.join(RES, 'ALL_RESULTS.csv'), index=False)
print(f"\n  Saved: ALL_RESULTS.csv")

# ── Figure: All-model 10-fold CV comparison ───────────────────────────────────
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
models  = df['model'].unique().tolist()
x       = np.arange(len(models))
colors  = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#C62828']

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, metric, label in zip(axes, metrics, labels):
    vals = [df[(df['model']==m) & (df['split']=='10-fold CV')][metric].values[0]
            if len(df[(df['model']==m) & (df['split']=='10-fold CV')]) else 0
            for m in models]
    bars = ax.bar(range(len(models)), vals, color=colors[:len(models)],
                  edgecolor='black', linewidth=0.6, alpha=0.88)
    ax.set_title(label, fontweight='bold', fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
    ax.set_ylim(max(0, min(vals) - 0.05), 1.01)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

plt.suptitle('All Models — 10-Fold Cross-Validation Performance\n'
             'DDoS-HTTP_Flood vs. BenignTraffic (CICIoT2023)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'all_models_cv_comparison.png'), dpi=150)
plt.close()
print("  Saved: all_models_cv_comparison.png")

# ── Figure: Training time comparison ─────────────────────────────────────────
cv_df = df[df['split'] == '10-fold CV']
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(cv_df['model'], cv_df['train_sec'],
              color=colors[:len(cv_df)], edgecolor='black', linewidth=0.6, alpha=0.85)
ax.set_title('Training Time — 10-Fold CV (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Seconds')
ax.set_xlabel('Model')
for bar, val in zip(bars, cv_df['train_sec']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}s', ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'training_time_comparison.png'), dpi=150)
plt.close()
print("  Saved: training_time_comparison.png")

# ── Figure: F1 score across all splits ────────────────────────────────────────
split_order = ['60/40', '70/30', '80/20', '10-fold CV']
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(split_order))
w = 0.17
for i, (m_name, color) in enumerate(zip(models, colors)):
    f1_vals = []
    for sp in split_order:
        row = df[(df['model'] == m_name) & (df['split'] == sp)]
        f1_vals.append(row['f1'].values[0] if len(row) else 0)
    ax.bar(x + i*w - 0.34, f1_vals, w, label=m_name, color=color, alpha=0.85,
           edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(split_order)
ax.set_ylim(0.90, 1.005)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
ax.set_ylabel('F1-Score')
ax.set_xlabel('Train / Test Split')
ax.set_title('F1-Score Across All Splits — All Models', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'f1_all_splits.png'), dpi=150)
plt.close()
print("  Saved: f1_all_splits.png")

# ── Print all figure paths ────────────────────────────────────────────────────
print("\n  ALL FIGURES:")
for f in sorted(os.listdir(FIGS)):
    print(f"    {FIGS}/{f}")

print("\n" + "=" * 65)
print("  ALL DONE. outputs/ is ready to submit.")
print("=" * 65)
