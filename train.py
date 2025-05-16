import torch
import torch.optim as optim
from preprocess import get_loaders
from model import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, test_loader, val_loader = get_loaders()
num_classes = len(train_loader.dataset.classes)


# Initialize model
# model = get_resnet50(num_classes=6).to(device)
# state_dict = torch.load("restnet3.pth", map_location="cpu")
# model.load_state_dict(state_dict)


model = CustomCNN(num_classes).to(device)
state_dict = torch.load("metal_optimized_model2.pth", map_location="cpu")
model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    train_acc = 100. * correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "custom1.pth")
