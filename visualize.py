import matplotlib.pyplot as plt
import os
import numpy as np
import re

def plot_confusion_matrix(confusion, label="Confusion_Matrix"):
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = range(len(confusion))  
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title=f'Confusion Matrix - {label}')

    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")

    fig.tight_layout()
    safe_label = re.sub(r'[\\/*?"<>|:]', '_', label)
    file_name = f"plots/{safe_label}_confusion_matrix.png"
    plt.savefig(file_name)
    print(f"Confusion matrix saved as {file_name}")
    plt.close()


def plot_all_training_histories(histories, compare_hidden=False):
    num_models = len(histories)
    plt.figure(figsize=(12, 5 * num_models))

    for i, (accuracies, losses, label) in enumerate(histories):
        epochs_range = range(1, len(accuracies) + 1)

        # Loss Plot
        plt.subplot(num_models, 2, i * 2 + 1)
        plt.plot(epochs_range, losses, label="Loss", color="red")
        plt.title(f"{label} - Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(num_models, 2, i * 2 + 2)
        plt.plot(epochs_range, accuracies, label="Accuracy", color="green")
        plt.title(f"{label} - Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)

    plt.tight_layout(h_pad=5.0)
    os.makedirs("plots", exist_ok=True)
    if compare_hidden:
        file_path = "plots/all_hidden_size_comparison.png"
    else:
        file_path = "plots/all_models_comparison.png"
    plt.savefig(file_path)
    print(f"Comparison plot saved as {file_path}")
    plt.show()


def plot_training_history(accuracies, losses, label, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    epochs_range = range(1, len(accuracies) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, label="Loss", color="red")
    plt.title(f"{label} - Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, label="Accuracy", color="green")
    plt.title(f"{label} - Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    safe_label = re.sub(r'[\\/*?:"<>|]', "_", label)
    file_name = f"{output_dir}/{safe_label.replace(' ', '_')}_training.png"

    plt.savefig(file_name)
    print(f"Training plot saved as {file_name}")
    plt.close()


def plot_accuracy_vs_hidden_size(test_accuracies, save_path="plots/accuracy_vs_size.png"):
    sizes, accuracies = zip(*test_accuracies)
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, accuracies, marker='o')
    plt.title("Test Accuracy vs. Hidden Layer Size")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()