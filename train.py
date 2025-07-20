import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import ASLClassifier
from utils import get_transforms, load_datasets

# Config
train_dir = 'data/asl_alphabet_train'
test_dir = 'data/asl_alphabet_test'
img_size = 128  # Increased for ResNet
batch_size = 32
epochs = 10
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Setup
train_tf, test_tf = get_transforms(img_size)
train_ds, test_ds, train_loader, test_loader = load_datasets(train_dir, test_dir, train_tf, test_tf, batch_size)

model = ASLClassifier(num_classes=len(train_ds.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training
best_test_acc = 0.0
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= total
    test_acc = 100 * correct / total
    scheduler.step()

    print(f"📊 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'sign_model.pth')  
        print("💾 Best model saved!")

print(f" Training complete. Best test accuracy: {best_test_acc:.2f}%")
