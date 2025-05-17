import torch
import torch.optim as optim
from preprocess import get_loaders
from model import *
from metrics import compute_accuracy, compute_metrics, plot_multiclass_roc, plot_confusion_matrix
from metrics import plot_metric_trends

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MINE_HARD = False
hard_ratio = 0.2

train_loader, _, val_loader = get_loaders()
num_classes = len(train_loader.dataset.classes)


# KEEP FOR TRAINING RESNET AND FURTHER TRAINING
model = get_resnet50(num_classes=6).to(device)
state_dict = torch.load("extra.pth", map_location="cpu")
model.load_state_dict(state_dict)


# model = CustomCNN(num_classes).to(device)
# state_dict = torch.load("custom1.pth", map_location="cpu")
# model.load_state_dict(state_dict)
if MINE_HARD:
    criterion = nn.CrossEntropyLoss(reduction='none')
else:
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
    train_preds = []
    train_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if MINE_HARD:
            losses = criterion(outputs, labels)  # shape [batch_size]

            # 2) pick indices of top-k hardest
            k = int(hard_ratio * losses.size(0))
            _, hard_idxs = torch.topk(losses, k, largest=True)

            # 3) re-compute loss on only hard examples (or zero out easy ones)
            hard_outputs = outputs[hard_idxs]
            hard_labels = labels[hard_idxs]
            loss = nn.CrossEntropyLoss()(hard_outputs, hard_labels)

        else:
            loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        train_preds.append(preds)
        train_labels.append(labels)

    all_train_preds = torch.cat(train_preds)
    all_train_labels = torch.cat(train_labels)
    train_acc = compute_accuracy(all_train_preds, all_train_labels)
    train_loss = running_loss
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

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

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    val_acc = 100. * val_correct / val_total
    f1, conf_matrix = compute_metrics(all_preds, all_labels)

    val_accuracies.append(val_acc)
    val_f1_scores.append(f1)
    print(f"[Epoch {epoch + 1}] Val Acc: {val_acc:.2f}% | F1: {f1:.4f}")


plot_metric_trends(train_losses, train_accuracies, val_accuracies, val_f1_scores)
plot_multiclass_roc(all_labels, all_logits, num_classes=num_classes, class_names=train_loader.dataset.classes)
plot_confusion_matrix(conf_matrix, class_names=train_loader.dataset.classes, normalize=True)
torch.save(model.state_dict(), "extra1.pth")
