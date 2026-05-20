"""
CSC 428 / Publication Extension
Step 8: Publication-Ready Summary
Author: M M Nishat

Merges all results (binary + multi-class + SHAP + robustness) and generates
the final tables and figures needed for the IEEE paper.
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

print("=" * 70)
print("  STEP 8 — PUBLICATION SUMMARY")
print("=" * 70)

# ── Load all results ───────────────────────────────────────────────────────────
binary  = pd.read_csv(os.path.join(RES, 'ALL_RESULTS.csv'))
mc      = pd.read_csv(os.path.join(RES, 'multiclass_results.csv'))
robust  = pd.read_csv(os.path.join(RES, 'robustness_noise.csv'))
ablation = pd.read_csv(os.path.join(RES, 'robustness_ablation.csv'))
shap_tbl = pd.read_csv(os.path.join(RES, 'shap_importance_table.csv'), index_col=0)
fam_drop = pd.read_csv(os.path.join(RES, 'robustness_per_family_drop.csv'), index_col=0)

# ── Table 1: Binary classification summary (10-fold CV) ───────────────────────
cv_binary = binary[binary['split'] == '10-fold CV'][
    ['model','accuracy','precision','recall','f1','roc_auc','train_sec']].copy()
cv_binary.columns = ['Model','Acc','Prec','Recall','F1','AUC','Time(s)']
print("\n  TABLE 1 — Binary Classification (10-Fold CV)")
print(cv_binary.round(4).to_string(index=False))

# ── Table 2: Multi-class summary ──────────────────────────────────────────────
mc_80 = mc[mc['level'].isin(['family','subtype'])].copy()
print("\n  TABLE 2 — Multi-Class Classification (80/20 Split, LightGBM)")
print(mc_80[['model','level','accuracy','f1_weighted','f1_macro','train_sec']
            ].round(4).to_string(index=False))

# ── Table 3: Robustness summary ───────────────────────────────────────────────
rob_global = robust[robust['scope'] == 'global'].copy()
print("\n  TABLE 3 — Robustness vs. Gaussian Noise (Global)")
print(rob_global[['sigma','accuracy','f1_weighted','acc_drop']].round(4).to_string(index=False))

# ── Table 4: Top-10 SHAP features ────────────────────────────────────────────
print("\n  TABLE 4 — Top 10 Global SHAP Features")
print(shap_tbl['global_mean'].head(10).round(4).to_string())

# ── Save master CSV ───────────────────────────────────────────────────────────
cv_binary.to_csv(os.path.join(RES, 'pub_table1_binary_cv.csv'), index=False)
mc_80.to_csv(os.path.join(RES, 'pub_table2_multiclass.csv'), index=False)
rob_global.to_csv(os.path.join(RES, 'pub_table3_robustness.csv'), index=False)
print("\n  Saved: pub_table1-3 CSVs")

# ── Figure: Combined 4-panel publication figure ───────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.35)

# Panel A: Binary CV F1 comparison
ax_a = fig.add_subplot(gs[0, 0])
models_b = cv_binary['Model'].tolist()
f1_b     = cv_binary['F1'].tolist()
colors_b = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#C62828']
bars = ax_a.bar(models_b, f1_b, color=colors_b, edgecolor='black', linewidth=0.6, alpha=0.88)
ax_a.set_ylim(0.97, 1.002)
ax_a.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
ax_a.set_title('(A) Binary Classification — 10-Fold CV F1', fontweight='bold', fontsize=10)
ax_a.set_xticklabels([m.replace(' ', '\n') for m in models_b], fontsize=8)
ax_a.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, f1_b):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
              f'{v:.4f}', ha='center', va='bottom', fontsize=7)

# Panel B: Multi-class family vs. subtype F1
ax_b = fig.add_subplot(gs[0, 1])
mc_pivot = mc[mc['level'].isin(['family','subtype'])].pivot(
    index='model', columns='level', values='f1_weighted')
x = np.arange(len(mc_pivot))
w = 0.35
ax_b.bar(x - w/2, mc_pivot['family'],  w, label='Family (7-class)',
         color='#1976D2', edgecolor='black', linewidth=0.6, alpha=0.88)
ax_b.bar(x + w/2, mc_pivot['subtype'], w, label='Subtype (34-class)',
         color='#F57C00', edgecolor='black', linewidth=0.6, alpha=0.88)
ax_b.set_xticks(x)
ax_b.set_xticklabels(mc_pivot.index, fontsize=9)
ax_b.set_ylim(0.94, 1.005)
ax_b.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
ax_b.set_title('(B) Multi-Class F1-Weighted (80/20 Split)', fontweight='bold', fontsize=10)
ax_b.legend(fontsize=8)
ax_b.grid(axis='y', alpha=0.3)

# Panel C: Robustness curve
ax_c = fig.add_subplot(gs[1, 0])
df_g = robust[robust['scope'] == 'global']
df_t = robust[robust['scope'] == 'targeted_top10_shap']
baseline_acc = df_g[df_g['sigma'] == 0.0]['accuracy'].values[0]
ax_c.plot(df_g['sigma'], df_g['accuracy'], marker='o', color='#1976D2',
          linewidth=2, label='Global noise', markersize=6)
ax_c.plot(df_t['sigma'], df_t['accuracy'], marker='s', color='#C62828',
          linewidth=2, linestyle='--', label='Targeted (top-10 SHAP)', markersize=6)
ax_c.axhline(baseline_acc, color='gray', linestyle=':', linewidth=1.2,
             label=f'Baseline ({baseline_acc:.4f})')
ax_c.set_xlabel('Noise σ'); ax_c.set_ylabel('Accuracy')
ax_c.set_title('(C) Adversarial Robustness — Accuracy vs. Noise σ',
               fontweight='bold', fontsize=10)
ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)

# Panel D: Top-10 SHAP features
ax_d = fig.add_subplot(gs[1, 1])
top10 = shap_tbl['global_mean'].head(10).sort_values()
top10.plot(kind='barh', ax=ax_d, color='#388E3C', edgecolor='black',
           linewidth=0.5, alpha=0.88)
ax_d.set_title('(D) Top 10 SHAP Features — Global Mean |SHAP|',
               fontweight='bold', fontsize=10)
ax_d.set_xlabel('Mean |SHAP Value|')
ax_d.grid(axis='x', alpha=0.3)

plt.suptitle(
    'Hierarchical IoT Intrusion Detection with SHAP Explainability\n'
    'and Adversarial Robustness Analysis — CICIoT2023',
    fontsize=14, fontweight='bold')
plt.savefig(os.path.join(FIGS, 'pub_main_figure.png'), dpi=200)
plt.close()
print("  Saved: pub_main_figure.png")

# ── Figure: Per-family robustness drop ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fam_sorted = fam_drop.sort_values('drop_%', ascending=False)
colors = ['#C62828' if v > 5 else '#1976D2' for v in fam_sorted['drop_%']]
bars = ax.bar(fam_sorted.index, fam_sorted['drop_%'],
              color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
for bar, v in zip(bars, fam_sorted['drop_%']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Accuracy Drop (%)', fontsize=11)
ax.set_xlabel('Attack Family', fontsize=11)
ax.set_title('Per-Family Accuracy Drop under Gaussian Noise (σ=0.5)',
             fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'pub_per_family_robustness.png'), dpi=150)
plt.close()
print("  Saved: pub_per_family_robustness.png")

# ── Print all output files ────────────────────────────────────────────────────
print("\n  ALL RESULT FILES:")
for f in sorted(os.listdir(RES)):
    print(f"    {f}")
print("\n  ALL FIGURES:")
for f in sorted(os.listdir(FIGS)):
    print(f"    {f}")

print("\n" + "=" * 70)
print("  STEP 8 COMPLETE — Publication outputs ready.")
print("=" * 70)
