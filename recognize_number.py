import os

import torch
import torch.nn as nn
import numpy as np
from visualize import plot_confusion_matrix, plot_all_training_histories, plot_training_history, plot_accuracy_vs_hidden_size
from utils import DigitNet, get_full_dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader


# ----- Training Logic -----
def train_model_on_split(dataset, train_ratio, model_path="model.pth", label="Model", seed=42, custom_model=None):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    torch.manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = custom_model if custom_model else DigitNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    confusion = np.zeros((10, 10), dtype=int)

    print(f"Training: {label}")
    accuracies = []
    losses = []

    for epoch in range(150):
        model.train()
        correct = 0
        total = 0
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        losses.append(total_loss)
        print(f"Epoch {epoch+1}, loss: {total_loss:.4f}, accuracy: {accuracy:.2f}%")

        if accuracy == 100:
            print("100% accuracy achieved! Stopping training.")
            break

    # Evaluation on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                if t != p:
                    confusion[t, p] += 1

    test_accuracy = 100 * correct / total
    print(f"Test accuracy for {label}: {test_accuracy:.2f}%")
    torch.save(model.state_dict(), model_path)
    plot_training_history(accuracies, losses, label)

    return model, accuracies, losses, label, confusion, test_accuracy


# ----- Main Execution -----
if __name__ == "__main__":
    full_dataset = get_full_dataset("painted_numbers")
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)

     # === EXPERIMENT 1: Impact of Training Set Size ===
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    histories_ratios = []

    print("\n==== EXPERIMENT 1: DIFFERENT TRAINING SET SIZES ====")
    for i, ratio in enumerate(ratios):
        model_path = os.path.join(models_dir, f"dataset_size_{int(ratio*100)}.pth")
        label = f"Train {int(ratio*100)}% / Test {int((1-ratio)*100)}%"
        model, acc, loss, lbl, confusion, test_acc = train_model_on_split(
            full_dataset, ratio, model_path, label
        )
        histories_ratios.append((acc, loss, lbl))
        print(f"Confusion matrix for: {label}")
        plot_confusion_matrix(confusion, label)

    plot_all_training_histories(histories_ratios)

    # === EXPERIMENT 2: Impact of Network Size (Number of Hidden Neurons) ===
    hidden_sizes = [16, 32, 64, 128, 256, 512]
    fixed_ratio = 0.5
    histories_sizes = []
    test_accuracies = []

    print("\n==== EXPERIMENT 2: DIFFERENT HIDDEN LAYER SIZES ====")
    for hidden_size in hidden_sizes:
        label = f"Hidden size: {hidden_size}"
        model_path = os.path.join(models_dir, f"hidden_size_{hidden_size}.pth")
        custom_model = DigitNet(hidden_size=hidden_size)

        model, acc, loss, lbl, confusion, test_acc = train_model_on_split(
            full_dataset, fixed_ratio, model_path, label, custom_model=custom_model
        )
        histories_sizes.append((acc, loss, lbl))
        test_accuracies.append((hidden_size, test_acc))
        print(f"Confusion matrix for: {label}")
        plot_confusion_matrix(confusion, label)

    plot_all_training_histories(histories_sizes, True)

    plot_accuracy_vs_hidden_size(test_accuracies)