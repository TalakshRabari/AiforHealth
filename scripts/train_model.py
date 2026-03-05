"""
train_model.py - Model Training with Leave-One-Participant-Out Cross-Validation
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[ERROR] PyTorch not found. Install it with: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.cnn_model import get_model

WINDOW_LENGTH = 960
N_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3


def load_dataset(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  -> {len(df)} windows, {df['participant_id'].nunique()} participants")
    print(f"  -> Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


def parse_raw_signals(df, window_length=WINDOW_LENGTH):
    n = len(df)
    X = np.zeros((n, N_CHANNELS, window_length), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for ch_idx, col in enumerate(['nasal_raw', 'thoracic_raw', 'spo2_raw']):
            try:
                vals = np.array([float(v) for v in str(row[col]).split(',')])
                if vals.std() > 1e-8:
                    vals = (vals - vals.mean()) / vals.std()
                length = min(len(vals), window_length)
                X[i, ch_idx, :length] = vals[:length]
            except Exception:
                pass
    return X


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = torch.argmax(model(X_batch.to(device)), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
    return np.array(all_true), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)*1.5),
                                    max(5, len(class_names)*1.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_lopo(csv_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(csv_path)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    class_names = list(le.classes_)
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}\n")

    print("Parsing raw signal data (this may take a moment)...")
    X_all = parse_raw_signals(df)
    y_all = df['label_encoded'].values.astype(np.int64)
    participants = df['participant_id'].values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    unique_participants = sorted(df['participant_id'].unique())
    fold_results = []
    all_true_combined = []
    all_pred_combined = []

    print("=" * 60)
    print("Leave-One-Participant-Out Cross-Validation")
    print("=" * 60)

    for fold_idx, test_participant in enumerate(unique_participants):
        print(f"\n-- Fold {fold_idx+1}/{len(unique_participants)}: Test = {test_participant} --")

        train_mask = participants != test_participant
        test_mask  = participants == test_participant

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        print(f"  Train: {X_train.shape[0]} windows | Test: {X_test.shape[0]} windows")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            batch_size=BATCH_SIZE, shuffle=False)

        model = get_model(n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * n_classes
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            y_true_val, y_pred_val = evaluate(model, test_loader, device)
            val_acc = accuracy_score(y_true_val, y_pred_val)
            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        model.load_state_dict(best_state)
        y_true, y_pred = evaluate(model, test_loader, device)
        all_true_combined.extend(y_true.tolist())
        all_pred_combined.extend(y_pred.tolist())

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        # Only report classes that appear in this fold
        fold_labels = sorted(set(y_true.tolist() + y_pred.tolist()))
        fold_names  = [class_names[i] for i in fold_labels]

        print(f"\n  Results for fold {fold_idx+1} (test={test_participant})")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred,
                                    labels=fold_labels,
                                    target_names=fold_names,
                                    zero_division=0))

        cm_path = os.path.join(output_dir, f'confusion_matrix_fold{fold_idx+1}_{test_participant}.png')
        plot_confusion_matrix(cm, class_names,
                              title=f'Confusion Matrix - Fold {fold_idx+1} (Test: {test_participant})',
                              save_path=cm_path)
        print(f"  -> Confusion matrix saved: {cm_path}")

        fold_results.append({
            'fold': fold_idx+1, 'test_participant': test_participant,
            'accuracy': acc, 'precision': prec, 'recall': rec
        })

        model_path = os.path.join(output_dir, f'model_fold{fold_idx+1}_{test_participant}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"  -> Model saved: {model_path}")

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (All 5 Folds)")
    print("=" * 60)

    results_df = pd.DataFrame(fold_results)
    print(results_df.to_string(index=False))
    print(f"\nMean Accuracy : {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
    print(f"Mean Precision: {results_df['precision'].mean():.4f} +/- {results_df['precision'].std():.4f}")
    print(f"Mean Recall   : {results_df['recall'].mean():.4f} +/- {results_df['recall'].std():.4f}")

    cm_overall = confusion_matrix(all_true_combined, all_pred_combined,
                                   labels=list(range(n_classes)))
    print(f"\nOverall Confusion Matrix:")
    print(pd.DataFrame(cm_overall, index=class_names, columns=class_names).to_string())

    overall_cm_path = os.path.join(output_dir, 'confusion_matrix_overall.png')
    plot_confusion_matrix(cm_overall, class_names,
                          title='Overall Confusion Matrix (LOPO CV)',
                          save_path=overall_cm_path)
    print(f"\n-> Overall confusion matrix saved: {overall_cm_path}")

    # Only report classes present across all folds
    all_labels = sorted(set(all_true_combined + all_pred_combined))
    all_names  = [class_names[i] for i in all_labels]
    print(f"\nOverall Classification Report:")
    print(classification_report(all_true_combined, all_pred_combined,
                                labels=all_labels,
                                target_names=all_names,
                                zero_division=0))

    results_path = os.path.join(output_dir, 'lopo_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"-> Results CSV saved: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', type=str, default='Dataset/breathing_dataset.csv')
    parser.add_argument('-output_dir', type=str, default='results')
    args = parser.parse_args()
    train_lopo(csv_path=args.dataset_path, output_dir=args.output_dir)