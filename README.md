
# CIFAR-10 Image Classification Using a Convolutional Neural Network

This repository explores image classification using a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras on the CIFAR-10 dataset. The project focuses on constructing and training a deep convolutional architecture while demonstrating core deep learning concepts such as convolutional feature extraction, pooling, regularization, and early stopping.

The objective is to demonstrate an end-to-end image classification workflow and practical understanding of deep learning techniques commonly applied to computer vision tasks.

---

## Project Overview

The project implements a complete deep learning workflow, including:

- Loading the CIFAR-10 dataset
- Building a multi-layer Convolutional Neural Network
- Convolution and Max Pooling operations
- Fully connected classification layers
- Dropout regularization
- L2 weight regularization
- Early stopping based on validation loss
- Model training and evaluation on unseen test data

The project demonstrates how convolutional neural networks progressively learn increasingly complex visual features for image classification tasks.

---

## Model Architecture

The implemented CNN consists of:

- Input layer for 32 × 32 RGB images
- Six Convolutional layers
- Three Max Pooling layers
- Flatten layer
- Two Dense fully connected layers
- Dropout regularization
- Softmax output layer for multi-class classification

The network is trained using:

- Adam optimizer
- Sparse categorical cross-entropy loss
- Early stopping with best-weight restoration

L2 regularization is applied throughout the convolutional and dense layers to reduce overfitting during training.

---

## Dataset

The project uses the CIFAR-10 dataset, a widely used benchmark dataset for image classification research and education.

The dataset contains:

- 60,000 colour images
- 10 object categories
- 32 × 32 pixel images
- Predefined training and testing splits

The dataset is loaded directly through the Keras datasets API.

---

## Training Strategy

The model is trained using:

- Batch size of 32
- 10 training epochs
- 20% validation split
- Early stopping based on validation loss

Early stopping restores the best-performing model weights to reduce overfitting during training.

---

## Limitations and Scope

This project is intentionally scoped as an educational and portfolio-oriented deep learning exercise. Current limitations include:

- No image normalization preprocessing
- No data augmentation
- Limited hyperparameter tuning
- No transfer learning
- No visualization of training history
- No confusion matrix or class-wise performance analysis

These limitations reflect a focus on understanding CNN architecture and training workflow rather than optimizing model performance for production deployment.

---

## Intended Audience and Use

This repository is intended to demonstrate:

- Practical understanding of convolutional neural networks
- Ability to design and train deep learning models
- Familiarity with TensorFlow/Keras workflows
- Knowledge of regularization and model evaluation techniques

This project is shared for educational and portfolio purposes.
