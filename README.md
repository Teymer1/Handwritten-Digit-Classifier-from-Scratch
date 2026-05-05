# Handwritten Digit Recognition Pipeline

A comprehensive machine learning project that covers the entire lifecycle of a computer vision task: from building a custom dataset via a GUI to conducting experimental analysis on neural network architectures.

## 🌟 Overview
This project demonstrates how to classify handwritten digits using a **Multi-Layer Perceptron (MLP)** built with **PyTorch**. Unlike standard MNIST tutorials, this pipeline includes a custom-built data collection tool and in-depth experiments on model scaling.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **GUI & Data Collection:** PyGame
* **Visualization:** Matplotlib
* **Data Processing:** NumPy & Pillow

## 📂 Project Structure
* `number_drawer.py` — GUI tool to draw and save custom digits (0-9).
* `recognize_number.py` — The core engine for training and running architecture experiments.
* `utils.py` — Contains the `DigitNet` model architecture and custom dataset loader.
* `visualize.py` — Utility for generating confusion matrices and training histories.
* `demo_images/` — A small sample set of hand-drawn digits for testing.
* `saved_models/` — Directory containing the best pre-trained weights (`final_model.pth`).
* `plots/` - Contains visualization of training and testing results, including accuracy vs. hidden layer size dependencies and comparative analysis across different models

## ✏️**Collect your own data:**
   Run `number_drawer.py`. Press **'S'** to save a digit, **'C'** to clear the screen, and use **0-9** keys to label your drawing.
   
## 🧪 Experiments & Analysis
I conducted two key experiments to evaluate the model's performance:

### 1. Training Set Size Impact
I analyzed how the volume of training data affects the final accuracy, testing ratios from **10% to 60%** of the collected dataset.

### 2. Hidden Layer Scaling
I evaluated the impact of network depth by testing hidden layer sizes of **16, 32, 64, 128, 256, and 512 neurons** to find the "sweet spot" for performance.

## 📊 Results
The best-performing model (128 hidden neurons) achieved high precision in digit classification.
   
## 🚀 How to Use
1. **Clone the repo:**
   ```bash
   git clone https://github.com/Teymer1/Handwritten-Digit-Classifier-from-Scratch
