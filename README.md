# cnn_image_classification
# 🖼️ CNN Image Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from **MNIST**, **Fashion-MNIST**, and **CIFAR-10** datasets. The model achieves high accuracy and supports training on GPUs for fast performance.

---

## 🚀 Features
✅ Train a CNN model on three popular datasets:  
- 🖊️ **MNIST** – Handwritten digits  
- 👕 **Fashion-MNIST** – Clothing items  
- 🚗 **CIFAR-10** – Real-world objects  

✅ Uses PyTorch-based **data augmentation** for CIFAR-10  
✅ Automatic saving of the best model based on validation accuracy  
✅ Supports training on **CPU** or **GPU (CUDA)**  

Create Virtual Environment (optional but recommended):

python -m venv venv
source venv/bin/activate      # On Linux/macOS
.\venv\Scripts\activate       # On Windows
pip install -r requirements.txt
python src/cnn_image_classification.py --dataset mnist --epochs 10 --batch_size 64
python src/cnn_image_classification.py --dataset fashion-mnist --epochs 10 --batch_size 64
python src/cnn_image_classification.py --dataset cifar10 --epochs 20 --batch_size 64
📊 Performance Metrics
✅ Loss – Cross Entropy Loss
✅ Optimizer – Adam Optimizer
✅ Learning Rate – 0.001

# Train on MNIST
python src/cnn_image_classification.py --dataset mnist --epochs 10 --batch_size 64

🧠 Future Work
✅ Try deeper CNN architectures
✅ Add additional datasets
✅ Implement learning rate scheduler

