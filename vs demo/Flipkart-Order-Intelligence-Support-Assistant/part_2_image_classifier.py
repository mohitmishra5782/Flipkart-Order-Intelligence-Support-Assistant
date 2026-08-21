import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision   import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np

def main():
    print("=== PART 2: TRANSFER LEARNING PRODUCT CLASSIFIER ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(3000))
    val_subset = Subset(test_dataset, range(500))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    model.train()
    for epoch in range(1):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Validation Accuracy: {correct / total * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/product_classifier.pt")
    print("Saved models/product_classifier.pt successfully!")

    os.makedirs("data/sample_images", exist_ok=True)
    raw_test = datasets.FashionMNIST(root="./data", train=False, download=True)
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    export_map = {0: "01_tshirt.png", 1: "02_trouser.png", 2: "03_pullover.png", 3: "04_dress.png", 8: "05_bag.png"}
    for idx in range(len(raw_test)):
        img, label = raw_test[idx]
        if label in export_map:
            filepath = os.path.join("data/sample_images", export_map[label])
            img.save(filepath)
            del export_map[label]

if __name__ == "__main__":
    main()