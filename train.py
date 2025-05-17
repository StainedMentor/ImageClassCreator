import torch
import torch.optim as optim
from preprocess import get_loaders
from model import *
from metrics import compute_accuracy, compute_metrics, plot_multiclass_roc, plot_confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, test_loader, val_loader = get_loaders()
num_classes = len(train_loader.dataset.classes)


# Initialize model
# model = get_resnet50(num_classes=6).to(device)
# state_dict = torch.load("restnet3.pth", map_location="cpu")
# model.load_state_dict(state_dict)


model = CustomCNN(num_classes).to(device)
# state_dict = torch.load("metal_optimized_model2.pth", map_location="cpu")
# model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 40

train_losses = []
train_accuracies = []
val_accuracies = []
val_f1_scores = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_acc = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        acc = compute_accuracy(preds, labels)
        train_acc += acc  # Track for epoch average

    train_losses.append(running_loss)
    train_accuracies.append(train_acc / len(train_loader))  # or just train_acc

    train_acc = 100. * correct / total
    print(f"[Epoch {epoch + 1}] Train Loss: {running_loss:.4f} | Train Acc: {train_acc / len(train_loader):.2f}%")

    model.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

            all_preds.append(preds)
            all_labels.append(labels)
            all_logits.append(outputs)
    # Concatenate all predictions
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)


    val_acc = 100. * val_correct / val_total
    f1, roc_auc, conf_matrix = compute_metrics(all_preds, all_labels, num_classes)

    val_accuracies.append(val_acc)
    val_f1_scores.append(f1)
    print(f"[Epoch {epoch + 1}] Val Acc: {val_acc:.2f}% | F1: {f1:.4f} | ROC AUC: {roc_auc if roc_auc else 'N/A'}")
    print("Confusion Matrix:\n", conf_matrix)

from metrics import plot_metric_trends

# After training loop ends
plot_metric_trends(train_losses, train_accuracies, val_accuracies, val_f1_scores)
plot_multiclass_roc(all_labels, all_logits, num_classes=num_classes, class_names=train_loader.dataset.classes)
plot_confusion_matrix(conf_matrix, class_names=train_loader.dataset.classes, normalize=True)
torch.save(model.state_dict(), "extra.pth")
