

# CIFAR-10 Image Classification with 81% Accuracy
This project uses a Deep Convolutional Neural Network (CNN) to classify 32x32 color images into 10 categories.

## 📊 Results
- **Training Accuracy:** ~80%
- **Validation Accuracy:** 81.7%
- **Techniques Used:** Batch Normalization, Dropout (0.2 to 0.5), Data Augmentation, Adam Optimizer.

## 🏗️ Architecture
The model consists of 3 main blocks:
1. Conv2D(32) -> BatchNorm -> Conv2D(32) -> MaxPool -> Dropout
2. Conv2D(64) -> BatchNorm -> Conv2D(64) -> MaxPool -> Dropout
3. Conv2D(128) -> BatchNorm -> MaxPool -> Dropout
4. Dense(512) -> Softmax(10)
