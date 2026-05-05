from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os


# ----- Custom Dataset -----
class CustomDigitDataset(Dataset):
    """
    A custom dataset loader that reads images from a folder.
    Expected filename format: '{label}_{index}.png' (e.g., '5_1.png').
    """

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Parse directory for image files and extract labels
        for fname in os.listdir(folder_path):
            if fname.endswith(".png") and "_" in fname:
                label = int(fname.split("_")[0])
                self.image_paths.append(os.path.join(folder_path, fname))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open image in grayscale mode ('L')
        image = Image.open(self.image_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# ----- Network Architecture -----
class DigitNet(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for digit classification.
    """

    def __init__(self, hidden_size=128):
        super(DigitNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        # Flatten the input tensor from (Batch, 1, 28, 28)
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_full_dataset(path):
    """
    Returns the dataset with the required image preprocessing transforms.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # Inverting the colors: PyGame draws white on black,
        # but standard MNIST models often expect black on white (or vice versa).
        transforms.Lambda(lambda x: 1.0 - x)
    ])
    return CustomDigitDataset(path, transform=transform)