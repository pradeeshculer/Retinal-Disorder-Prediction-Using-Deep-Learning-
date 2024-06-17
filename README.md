# Retinal Disorder Prediction Project

## Table of Contents
1. [Project Description](#1-project-description)
2. [Dataset Information](#2-dataset-information)
   - [Source](#source)
   - [Description](#description)
   - [Preprocessing Steps](#preprocessing-steps)
3. [Model Descriptions](#3-model-descriptions)
   - [VGG19](#vgg19)
   - [Xception](#xception)
   - [Custom CNN](#custom-cnn)
4. [Performance Metrics](#4-performance-metrics)
   - [VGG19](#vgg19-performance)
   - [Xception](#xception-performance)
   - [Custom CNN](#cnn-performance)
5. [Training Details](#5-training-details)
   - [Hardware and Software Specifications](#hardware-and-software-specifications)
   - [Training Parameters](#training-parameters)
   - [Data Augmentation and Regularization](#data-augmentation-and-regularization)
6. [Instructions for Use](#6-instructions-for-use)
   - [Requirements](#requirements)
7. [Web Application](#7-web-application)
   - [Frontend](#frontend)
   - [Backend](#backend)
8. [Conclusion](#8-conclusion)

## 1. Project Description
This project aims to develop a machine learning model capable of predicting different types of retinal disorders based on Optical Coherence Tomography (OCT) images. Accurate and early diagnosis of retinal disorders such as Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen can significantly enhance treatment outcomes and patient care. This project leverages deep learning architectures to achieve high accuracy in classifying retinal images into four categories: CNV, DME, Drusen, and Normal.

## 2. Dataset Information
### Source
The dataset used in this project is sourced from Mendeley.

### Description
- **Type of Images**: OCT (Optical Coherence Tomography)
- **Number of Images**:
  - Healthy: 26,315 images
  - Choroidal Neovascularization (CNV): 37,205 images
  - Diabetic Macular Edema (DME): 11,348 images
  - Drusen: 8,616 images

### Preprocessing Steps
1. Conversion of white pixels to black pixels in OCT scans.
2. Consistent cropping of the images to a size of 128x128 pixels.

These preprocessing steps ensure homogeneity in input size and accelerate the convergence of the models during training.

## 3. Model Descriptions
### VGG19
- **Introduction**: VGG19 is a deep convolutional neural network developed by the Visual Geometry Group at the University of Oxford.
- **Architecture Details**: VGG19 consists of 19 layers, including convolutional layers, max-pooling layers, and fully connected layers. It is known for its uniform architecture and small receptive fields.
- **Why Chosen**: VGG19's deep architecture and pre-trained weights on ImageNet provide a robust feature extraction mechanism, making it suitable for complex image classification tasks.

### Xception
- **Introduction**: Xception (Extreme Inception) is a deep convolutional neural network that leverages depthwise separable convolutions.
- **Architecture Details**: Xception has a 71-layer deep architecture with depthwise separable convolutions that improve efficiency and performance.
- **Why Chosen**: Xception's architecture allows for efficient training and improved accuracy by reducing the number of parameters while maintaining high performance.

### Custom CNN
- **Architecture Details**:
  - **Convolutional Layers**: 5 layers with increasing filter sizes (16, 32, 64, 128, 256), each followed by max-pooling layers.
  - **Flatten Layer**: Flattens the 2D feature maps to 1D feature vectors.
  - **Dropout Layer**: Dropout rate of 0.3 to prevent overfitting.
  - **Dense Layers**: A dense layer with 256 units and ReLU activation, followed by an output layer with 4 units and softmax activation for multi-class classification.
- **Why Designed This Way**: The custom CNN was designed to capture features at various levels of abstraction while minimizing overfitting and computational complexity.

## 4. Performance Metrics
The models were evaluated based on their accuracy in classifying the retinal images.

### VGG19 Performance
- **Overall Accuracy**: 87.15%

**Table 4.1: Test Results of VGG19**

| Disease | Precision | Recall | F1-score | Accuracy |
| ------- | --------- | ------ | -------- | -------- |
| CNV     | 0.98      | 0.96   | 0.97     | 91.51%   |
| DME     | 0.92      | 0.88   | 0.90     | 88.43%   |
| DRUSEN  | 0.78      | 0.83   | 0.80     | 82.90%   |
| NORMAL  | 0.85      | 0.85   | 0.85     | 84.57%   |

### Xception Performance
- **Overall Accuracy**: 85.89%

**Table 4.2: Test Results of Xception**

| Disease | Precision | Recall | F1-score | Accuracy |
| ------- | --------- | ------ | -------- | -------- |
| CNV     | 0.92      | 0.91   | 0.91     | 90.61%   |
| DME     | 0.87      | 0.88   | 0.87     | 87.53%   |
| DRUSEN  | 0.83      | 0.78   | 0.80     | 77.89%   |
| NORMAL  | 0.82      | 0.88   | 0.85     | 87.53%   |

### Custom CNN Performance
- **Overall Accuracy**: 95.47%

**Table 4.3: Test Results of CNN**

| Disease | Precision | Recall | F1-score | Accuracy |
| ------- | --------- | ------ | -------- | -------- |
| CNV     | 0.98      | 0.96   | 0.97     | 95.75%   |
| DME     | 0.95      | 0.97   | 0.96     | 96.78%   |
| DRUSEN  | 0.94      | 0.96   | 0.95     | 95.50%   |
| NORMAL  | 0.94      | 0.94   | 0.94     | 93.83%   |

## 5. Training Details
### Hardware and Software Specifications
- **Hardware**: High-performance GPU (NVIDIA)
- **Software**: TensorFlow, Keras, Python

### Training Parameters
- **Epochs**: 
  - VGG19: 20 epochs
  - Xception: 25 epochs
  - Custom CNN: 20 epochs, extended to 100 epochs with checkpoints
- **Batch Size**: 64
- **Learning Rate**: 
  - Optimizer: Adam
  - Learning Rate: 0.001

### Data Augmentation and Regularization
- **Data Augmentation**: Rescaling of images by 1./255.
- **Regularization Techniques**: Dropout layer with a rate of 0.3 to prevent overfitting.

## 6. Instructions for Use
### Requirements
- TensorFlow
- Keras
- Flask
- Numpy
- Matplotlib
- Pandas

## 7. Web Application

A Flask-based web application was developed to allow users to upload OCT images and get predictions. The application provides information about the predicted disease category, signs, symptoms, and potential cures.

### Frontend

The frontend is a simple HTML/CSS interface that allows users to upload images and view predictions.

### Backend 

The backend is powered by Flask and serves the model predictions. The model used for the web application is the custom CNN due to its high accuracy.

## 8. Conclusion

This project underscores the effectiveness of deep learning in enhancing diagnostic capabilities for retinal disorders through OCT imaging. The utilization of tailored CNN architectures, particularly the custom CNN model, showcases superior performance metrics compared to established architectures like VGG19 and Xception. The findings not only validate the efficacy of deep learning in medical image analysis but also emphasize the potential for implementing such models in clinical settings to aid ophthalmologists in timely and precise diagnosis.



