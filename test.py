import torch
import matplotlib.pyplot as plt

from metrics import compute_metrics, compute_accuracy
from model import CustomCNN, get_resnet50
from preprocess import get_loaders
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
_, test_loader, _ = get_loaders()
num_classes = len(test_loader.dataset.classes)

baseline = CustomCNN(num_classes).to(device)
baseline.load_state_dict(torch.load("custom1.pth", map_location="cpu"))
baseline.eval()

hard = CustomCNN(num_classes).to(device)
hard.load_state_dict(torch.load("custom2.pth", map_location="cpu"))
hard.eval()

resnet = get_resnet50(num_classes).to(device)
resnet.load_state_dict(torch.load("extra.pth", map_location="cpu"))
resnet.eval()

models = [
    ("Baseline CNN", baseline),
    ("Hard-mined CNN", hard),
    ("ResNet50", resnet)
]

model_names = []
accuracies = []
f1_scores = []
roc_labels_list = []
roc_logits_list = []
conf_matrices_list = []

for name, model in models:
    model_names.append(name)
    val_correct = 0
    val_total = 0
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

            all_preds.append(preds)
            all_labels.append(labels)
            all_logits.append(outputs)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    acc = compute_accuracy(all_preds, all_labels)
    f1, conf_matrix = compute_metrics(all_preds, all_labels)

    accuracies.append(acc)
    f1_scores.append(f1)
    roc_labels_list.append(all_labels)
    roc_logits_list.append(all_logits)
    conf_matrices_list.append(conf_matrix)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(model_names, accuracies)
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.grid(axis="y")

plt.subplot(1, 2, 2)
plt.bar(model_names, f1_scores)
plt.ylabel("F1 Score")
plt.title("Test F1 Score Comparison")
plt.grid(axis="y")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(model_names), figsize=(15, 5))
for idx, name in enumerate(model_names):
    labels = roc_labels_list[idx].cpu().numpy()
    logits = roc_logits_list[idx]
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    labels_one_hot = label_binarize(labels, classes=range(num_classes))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        axes[idx].plot(fpr, tpr, label=f"{test_loader.dataset.classes[i]} (AUC = {auc_score:.2f})")
    axes[idx].plot([0,1], [0,1], 'k--', label='Random')
    axes[idx].set_title(f"{name} ROC")
    axes[idx].set_xlabel("False Positive Rate")
    axes[idx].set_ylabel("True Positive Rate")
    axes[idx].legend(loc='lower right')
    axes[idx].grid(True)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(model_names), figsize=(15, 5))
for idx, name in enumerate(model_names):
    cm = conf_matrices_list[idx]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes,
                ax=axes[idx], cmap="Blues")
    axes[idx].set_title(f"{name} Confusion Matrix")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")
plt.tight_layout()
plt.show()