# Auto Insurance Claims Prediction

This project implements a neural network model to predict auto insurance claims based on input features. The model uses TensorFlow with custom optimization and feature transformation techniques to improve prediction accuracy.

## Overview

The model predicts insurance claim amounts using a simple feed-forward neural network architecture. It includes:
- Power transformation for feature normalization
- Custom AdamW optimizer with weight decay
- Early stopping to prevent overfitting
- Mean Absolute Error (MAE) evaluation

## Dataset

The model uses the `auto-insurance.csv` dataset which contains:
- Input feature: Claims frequency
- Target variable: Total payment for all claims in thousands of Swedish Kronor

## Requirements

```
pandas
scikit-learn
tensorflow
tensorflow-addons
matplotlib
```

## Installation

```bash
git clone https://github.com/nandanmn/Virtual_presence.git
cd Virtual_presence
pip install -r requirements.txt
```

## Model Architecture

The neural network consists of:
- Input layer: 1 feature
- Hidden layer 1: 10 neurons with ReLU activation
- Hidden layer 2: 8 neurons with ReLU activation
- Output layer: 1 neuron (linear activation)

## Feature Engineering

- Power transformation is applied to both input and output variables
- Data is split into training (67%) and testing (33%) sets

## Training

The model is trained with:
- AdamW optimizer (learning rate = 0.01, weight decay = 0.01)
- Mean Squared Error loss function
- Early stopping with 10 epochs patience
- Batch size of 4
- Maximum 300 epochs

## Results

The model achieves a Mean Absolute Error of approximately 39.065 on the test set.

## Usage

```python
from pandas import read_csv
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load and preprocess data
df = read_csv('auto-insurance.csv', header=None)
X, y = df.values[:, :-1], df.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

# Train model
model.fit(X_train, y_train, epochs=300, batch_size=4, 
          callbacks=[early_stopping], validation_data=(X_test,y_test))
```
Thank You
