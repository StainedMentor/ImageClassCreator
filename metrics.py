import seaborn as sns
import torch
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def compute_accuracy(preds, labels):
    correct = preds.eq(labels).sum().item()
    total = labels.size(0)
    return 100. * correct / total

def compute_metrics(preds, labels):
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    f1 = f1_score(labels_np, preds_np, average='weighted')
    conf_matrix = confusion_matrix(labels_np, preds_np)

    return f1, conf_matrix



def plot_multiclass_roc(labels, logits, num_classes, class_names=None):
    labels = labels.cpu().numpy()
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    labels_one_hot = label_binarize(labels, classes=range(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_one_hot[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        label = f"Class {i}" if class_names is None else class_names[i]
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metric_trends(train_losses, train_accs, val_accs, f1_scores):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, f1_scores, label='F1 Score', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names=None, normalize=False):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()