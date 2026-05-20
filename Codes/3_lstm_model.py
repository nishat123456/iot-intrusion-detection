"""
CSC 428 - Machine Learning in Cybersecurity
Final Project — Step 3: LSTM Model (Advanced / Extension)
Author: M M Nishat

LSTM treats each of the N features as a time-step of length 1,
giving input shape (samples, n_features, 1). This is the standard
approach used in the IoT-IDS literature for applying RNNs to
tabular network-flow data.
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
RES    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

# ── Load data ─────────────────────────────────────────────────────────────────
splits        = joblib.load(os.path.join(RES, 'splits.pkl'))
feature_names = joblib.load(os.path.join(RES, 'feature_names.pkl'))
X_all = pd.read_csv(os.path.join(RES, 'X_scaled.csv')).values
y_all = pd.read_csv(os.path.join(RES, 'y.csv')).squeeze().values

n_features = X_all.shape[1]

print("=" * 65)
print("  STEP 3 — LSTM MODEL")
print("=" * 65)
print(f"  Features (time-steps): {n_features}")
print(f"  Input shape: (samples, {n_features}, 1)")

# ── Model builder ──────────────────────────────────────────────────────────────
def build_lstm(n_features):
    model = Sequential([
        LSTM(64, input_shape=(n_features, 1), return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

print("\n  LSTM Architecture:")
build_lstm(n_features).summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, min_delta=1e-4,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0),
]

# ── Evaluate across 3 splits ─────────────────────────────────────────────────
lstm_results = []
history_80_20 = None

SPLIT_LABELS = {'60_40': '60/40', '70_30': '70/30', '80_20': '80/20'}

for tag, (X_tr, X_te, y_tr, y_te) in splits.items():
    split_label = SPLIT_LABELS[tag]
    print(f"\n  ── Split {split_label} ──")

    X_tr_3d = X_tr.values.reshape(-1, n_features, 1)
    X_te_3d = X_te.values.reshape(-1, n_features, 1)
    y_tr_np = y_tr.values
    y_te_np = y_te.values

    t0    = time.time()
    tf.random.set_seed(42)
    model = build_lstm(n_features)
    hist  = model.fit(
        X_tr_3d, y_tr_np,
        validation_split=0.10,
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = round(time.time() - t0, 2)
    epochs_ran = len(hist.history['loss'])

    y_prob = model.predict(X_te_3d, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    row = {
        'model':     'LSTM',
        'split':     split_label,
        'accuracy':  round(accuracy_score(y_te_np, y_pred), 4),
        'precision': round(precision_score(y_te_np, y_pred), 4),
        'recall':    round(recall_score(y_te_np, y_pred), 4),
        'f1':        round(f1_score(y_te_np, y_pred), 4),
        'roc_auc':   round(roc_auc_score(y_te_np, y_prob), 4),
        'train_sec': train_time,
        'epochs':    epochs_ran,
    }
    lstm_results.append(row)

    print(f"    Epochs: {epochs_ran}  Time: {train_time}s")
    print(f"    Acc={row['accuracy']}  F1={row['f1']}  AUC={row['roc_auc']}")

    if tag == '80_20':
        history_80_20 = hist
        cm_lstm = confusion_matrix(y_te_np, y_pred)
        fpr_l, tpr_l, _ = roc_curve(y_te_np, y_prob)
        lstm_roc = (fpr_l, tpr_l, row['roc_auc'])
        model.save(os.path.join(MODELS, 'lstm_80_20.h5'))

# ── 10-Fold CV (lightweight: 5 epochs, no early stopping for speed) ───────────
print("\n  ── 10-Fold Cross-Validation ──")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
X_3d = X_all.reshape(-1, n_features, 1)
t0 = time.time()

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_3d, y_all), 1):
    m = build_lstm(n_features)
    m.fit(X_3d[tr_idx], y_all[tr_idx],
          epochs=10, batch_size=256, verbose=0,
          validation_split=0.0)
    y_p = m.predict(X_3d[te_idx], verbose=0).ravel()
    y_pr = (y_p >= 0.5).astype(int)
    fold_metrics['accuracy'].append(accuracy_score(y_all[te_idx], y_pr))
    fold_metrics['precision'].append(precision_score(y_all[te_idx], y_pr))
    fold_metrics['recall'].append(recall_score(y_all[te_idx], y_pr))
    fold_metrics['f1'].append(f1_score(y_all[te_idx], y_pr))
    fold_metrics['roc_auc'].append(roc_auc_score(y_all[te_idx], y_p))
    print(f"    Fold {fold:2d}: Acc={fold_metrics['accuracy'][-1]:.4f}  "
          f"F1={fold_metrics['f1'][-1]:.4f}")

cv_time = round(time.time() - t0, 2)
cv_row = {
    'model':     'LSTM',
    'split':     '10-fold CV',
    'accuracy':  round(np.mean(fold_metrics['accuracy']), 4),
    'precision': round(np.mean(fold_metrics['precision']), 4),
    'recall':    round(np.mean(fold_metrics['recall']), 4),
    'f1':        round(np.mean(fold_metrics['f1']), 4),
    'roc_auc':   round(np.mean(fold_metrics['roc_auc']), 4),
    'train_sec': cv_time,
    'epochs':    10,
}
lstm_results.append(cv_row)
print(f"    CV mean — Acc={cv_row['accuracy']}  F1={cv_row['f1']}  AUC={cv_row['roc_auc']}")

# ── Save LSTM results ─────────────────────────────────────────────────────────
df_lstm = pd.DataFrame(lstm_results)
df_lstm.to_csv(os.path.join(RES, 'lstm_results.csv'), index=False)

# ── Figure: Training history (80/20) ─────────────────────────────────────────
if history_80_20:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history_80_20.history['loss'],     label='Train Loss', color='#1976D2')
    ax1.plot(history_80_20.history['val_loss'], label='Val Loss',   color='#F44336', linestyle='--')
    ax1.set_title('LSTM — Training & Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history_80_20.history['accuracy'],     label='Train Acc', color='#1976D2')
    ax2.plot(history_80_20.history['val_accuracy'], label='Val Acc',   color='#F44336', linestyle='--')
    ax2.set_title('LSTM — Training & Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle('LSTM Training History (80/20 split)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'lstm_training_history.png'), dpi=150)
    plt.close()
    print("\n  Saved: lstm_training_history.png")

# ── Figure: LSTM Confusion Matrix (80/20) ────────────────────────────────────
import seaborn as sns
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Benign', 'DDoS'], yticklabels=['Benign', 'DDoS'])
ax.set_title('LSTM Confusion Matrix — 80/20 Split', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'lstm_confusion_matrix.png'), dpi=150)
plt.close()
print("  Saved: lstm_confusion_matrix.png")

print("\n" + "=" * 65)
print("  LSTM COMPLETE. Run 4_final_summary.py next.")
print("=" * 65)
