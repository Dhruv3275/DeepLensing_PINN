# Physics Guided Machine Learning on Real Lensing Images Using Resnet-18, Resnet-10 and Physics Informed Neural Network (PINN)
## Overview

This repository contains the implementation of two models for classifying images of gravitational lenses using PyTorch. The first model employs a standard ResNet-18 architecture, while the second model enhances the ResNet-18 with a physics-informed neural network (PINN) that incorporates the gravitational lensing equation.

## Common Test: Multi-Class Classification

### Task

Build a model for classifying images into lenses using PyTorch or Keras. The model should be trained and validated using a suitable approach to achieve high accuracy.

### Conversion

The dataset which was given to us was in .npy format for both train and validation which if we tried to extract directly and used as our input and tried to run it resulted in getting not proper result so we first convert it into .png files,with the help of os and PIL and then with help of Transformers we were able to properly use it for our training purpose.

### Notebook

You can find the conversion notebook in the following Jupyter notebook:
[Conversion of Data](https://github.com/Dhruv3275/DeepLensing_PINN/blob/main/Conversion%20of%20Data.ipynb)

### Approach-1

I utilized ResNet-18 for this task due to its ability to capture intricate patterns and features in the data, leading to better generalization and higher accuracy. The model was trained using 10-fold cross-validation, with each fold consisting of 3 epochs on a dataset of 30,000 training images and 7,500 test images.

### Results

The model achieved an ROC-AUC score of 0.99 on the test data.


<img src="ROC-AUC Curves/Classification/ResNet 18.png" alt="ROC-AUC" width="400"/>


### Notebook

You can find the detailed implementation and results of this approach in the following Jupyter notebook: [ResNet-18 Approach](https://github.com/Dhruv3275/DeepLensing_PINN/blob/main/ResNet%2018%20Approach%20Classification.ipynb)

### Approach-2

I utilized ResNet-10 for this task due to its ability to capture intricate patterns and generalize well, achieving competitive accuracy. The model was trained using 10-fold cross-validation, with each fold running for 3 epochs on a dataset of 30,000 training images and 7,500 test images. This approach helps evaluate the model's robustness while exploring architectures that maintain high accuracy with fewer parameters, improving computational efficiency.

### Results

The model achieved an ROC-AUC score of 0.99 on the test data.


<img src="ROC-AUC Curves/Classification/ResNet 10.png" alt="ROC-AUC" width="400"/>


### Notebook

You can find the detailed implementation and results of this approach in the following Jupyter notebook: [ResNet-10 Approach]
[ResNet 10 Approach Classification.pdf](https://github.com/Dhruv3275/DeepLensing_PINN/blob/main/ResNet%2010%20Approach%20Classification.ipynb)

## Specific Test V: Physics-Guided ML

### Task

Build a model for classifying images into lenses using PyTorch or Keras. The architecture should take the form of a physics-informed neural network (PINN) that incorporates the gravitational lensing equation to improve network performance over the common test results.

### Conversion

The dataset which was given to us was in .npy format for both train and validation which if we tried to extract directly and used as our input and tried to run it resulted in getting not proper result so we first convert it into .png files,with the help of os and PIL and then with help of Transformers we were able to properly use it for our training purpose.

### Notebook

You can find the conversion notebook in the following Jupyter notebook:
[Conversion of Data](https://github.com/Dhruv3275/DeepLensing_PINN/blob/main/Conversion%20of%20Data.ipynb)


### Approach 1

For classifying gravitational lenses into three types (no lensing, vortex, and halo substructure), I incorporated the lens equation, which describes how light is bent by the gravitational field of a massive object. The mass distribution of the lensing object is assumed to follow a Singular Isothermal Sphere (SIS) model, with a proportionality parameter \( k \) to correct potential distortions.

#### Implementation Steps

1. Define the lens equation:
    β=θ−α
    where β is the apparent position of the source, θ is the observed position, and α is the deflection angle.
2. Incorporate the mass distribution due to galaxies and dark matter:
    β+cX=θ−kr^2

3. Utilize feature vectors θ and k from ResNet-18.
4. Apply three neural layers on the resulting vector to extract features for lens classification.

### Results

The model achieved an ROC-AUC score of 0.97 on the test data.


<img src="ROC-AUC Curves/PINNs/ResNet 18.png" alt="ROC-AUC" width="400"/>


### Notebook

You can find the detailed implementation and results of this approach in the following Jupyter notebook: 
[PINN Approach 1](https://github.com/Dhruv3275/DeepLensing_PINN/blob/main/ResNet%2018%20Approach%20PINNs.ipynb)

---

