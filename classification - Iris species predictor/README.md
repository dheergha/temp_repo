# PyTorch Classification: Iris Species Predictor

This directory contains the PyTorch implementation of a multiclass classification task, classifying species of Iris flowers using flower measurements.

## 1. Dataset Description
The model utilizes the well-known **Iris Dataset**. It deals with classifying 150 instances of Iris flowers into one of three species: `setosa`, `versicolor`, or `virginica`.
* **Features**: sepal length, sepal width, petal length, and petal width (4 continuous numerical variables).
* **Target**: Categorical variable corresponding to the species (3 classes).

## 2. Preprocessing Steps
* **Data Retrieval**: Loaded natively using `sklearn.datasets.load_iris` to avoid dependency on an external `.csv`.
* **Feature Scaling**: Numerical measurements are uniformly scaled using `StandardScaler` to boost model convergence rates.
* **Data Splitting**: We applied a standard 80/20 train/test split utilizing `train_test_split`.
* **Tensor Conversion**: The numeric arrays and labels were transitioned to PyTorch Tensors (`torch.float32` and `torch.long` respectively); subsequently handled by `TensorDataset` and `DataLoader`.

## 3. Model Architecture Details
The script builds a Feedforward Neural Network (`ClassificationNN`) tailored for multiclass scenarios.
* **Input Layer**: Size 4 (for our 4 distinct Iris dataset features).
* **Hidden Layer 1**: Linear(4, 32) + ReLU Activation
* **Hidden Layer 2**: Linear(32, 16) + ReLU Activation
* **Output Layer**: Linear(16, 3) representing the logits for our 3 unique outputs. (Softmax application is omitted locally as it acts implicitly through PyTorch's `CrossEntropyLoss`).

## 4. Evaluation Metrics and Result Discussion
The network was optimized using the Adam optimizer (`lr=0.01`) and PyTorch's CrossEntropyLoss. Over `100` epochs, the loss gracefully minimized towards convergence.
Performance measures monitored directly on out-of-bag test set features:
* **Accuracy Score**: Commonly strikes extremely close identically 100% due to the simplicity of the test dataset and cleanly separated embeddings.
* **Summary Classification Report**: Precisely documents *Precision, Recall, and F1-Scores* across all categorical labels evaluated. 
* **Takeaway**: Strong separation mapping by the non-linear Multi-layer perceptron.

## 5. Instructions to Run the Project
1. Open up an adequate Python environment.
2. Confirm library prerequisites:
   ```bash
   pip install torch scikit-learn pandas matplotlib numpy
   ```
3. To enact evaluation, merely fire:
   ```bash
   python train_classification.py
   ```
4. Output plots (`training_loss.png`) and model parameter weights (`model.pth`) are deposited straight to this folder cleanly confirming operability.