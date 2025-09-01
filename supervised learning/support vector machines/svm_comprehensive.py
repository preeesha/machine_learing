"""
Support Vector Machines (SVM) - Comprehensive Guide
==================================================

This file provides a complete understanding of Support Vector Machines including:
- Mathematical foundations
- Different types of SVM
- Kernel functions
- Practical implementation with scikit-learn
- Visualization examples
- Real-world dataset applications

Author: Learning Class
Date: 2024
"""

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           mean_squared_error, r2_score, roc_curve, auc)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# 1. INTRODUCTION TO SUPPORT VECTOR MACHINES
# =============================================================================

"""
SUPPORT VECTOR MACHINES (SVM) - CORE CONCEPTS
============================================

What is SVM?
------------
- A supervised learning algorithm used for classification, regression, and outlier detection
- Finds an optimal hyperplane that best separates data points of different classes
- Particularly effective in high-dimensional spaces

Key Components:
1. Hyperplane: A decision boundary that separates data points
2. Support Vectors: Data points closest to the hyperplane that influence its position
3. Margin: The distance between the hyperplane and nearest data points from each class
4. Maximum Margin: SVM aims to maximize this margin for better generalization

Mathematical Foundation:
- Linear SVM finds a hyperplane: w^T * x + b = 0
- Classification: if w^T * x + b > 0 → Class +1, else Class -1
- Objective: Minimize (1/2)||w||^2 subject to y_i(w^T * x_i + b) ≥ 1
"""

# =============================================================================
# 2. LINEAR SVM - LINEARLY SEPARABLE DATA
# =============================================================================

def create_linear_separable_data():
    """
    Create linearly separable data for demonstrating basic SVM concepts
    
    Returns:
        X: Feature matrix (100 samples, 2 features)
        y: Target labels (binary: +1, -1)
    """
    np.random.seed(42)
    
    # Generate data for class 1 (blue points)
    class1_x = np.random.normal(2, 1, 50)
    class1_y = np.random.normal(2, 1, 50)
    
    # Generate data for class 2 (red points)
    class2_x = np.random.normal(6, 1, 50)
    class2_y = np.random.normal(6, 1, 50)
    
    # Combine data
    X = np.vstack([np.column_stack([class1_x, class1_y]), 
                   np.column_stack([class2_x, class2_y])])
    y = np.hstack([np.ones(50), -np.ones(50)])
    
    return X, y

def demonstrate_linear_svm():
    """
    Demonstrate Linear SVM with synthetic linearly separable data
    Shows decision boundary, support vectors, and margin
    """
    print("=== LINEAR SVM DEMONSTRATION ===")
    
    # Create data
    X_linear, y_linear = create_linear_separable_data()
    
    # Train linear SVM
    linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
    linear_svm.fit(X_linear, y_linear)
    
    # Get support vectors
    support_vectors = linear_svm.support_vectors_
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(X_linear[y_linear == 1][:, 0], X_linear[y_linear == 1][:, 1], 
                c='blue', label='Class 1 (+1)', s=50, alpha=0.7)
    plt.scatter(X_linear[y_linear == -1][:, 0], X_linear[y_linear == -1][:, 1], 
                c='red', label='Class -1', s=50, alpha=0.7)
    
    # Plot support vectors (highlighted)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                c='green', s=200, alpha=0.8, linewidth=2, 
                edgecolor='black', label='Support Vectors')
    
    # Plot decision boundary and margin
    x_min, x_max = X_linear[:, 0].min() - 1, X_linear[:, 0].max() + 1
    y_min, y_max = X_linear[:, 1].min() - 1, X_linear[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear SVM: Decision Boundary and Support Vectors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print results
    print(f"Number of support vectors: {len(support_vectors)}")
    print(f"Training accuracy: {linear_svm.score(X_linear, y_linear):.4f}")
    print(f"Model parameters - w: {linear_svm.coef_[0]}, b: {linear_svm.intercept_[0]:.4f}")

# =============================================================================
# 3. NON-LINEAR SVM - KERNEL TRICKS
# =============================================================================

"""
KERNEL FUNCTIONS IN SVM
=======================

Why Kernels?
------------
- Linear SVM can only handle linearly separable data
- Real-world data is often non-linearly separable
- Kernel functions transform data into higher-dimensional space where it becomes linearly separable

Common Kernel Functions:
1. Linear: K(x, y) = x^T * y
2. Polynomial: K(x, y) = (γ * x^T * y + r)^d
3. RBF (Gaussian): K(x, y) = exp(-γ * ||x - y||^2)
4. Sigmoid: K(x, y) = tanh(γ * x^T * y + r)

Parameters:
- C: Regularization parameter (controls trade-off between margin and misclassification)
- γ (gamma): Kernel coefficient for RBF, polynomial, and sigmoid kernels
- d: Degree of polynomial kernel
- r: Constant term in polynomial and sigmoid kernels
"""

def create_non_linear_data():
    """
    Create non-linearly separable data (XOR-like pattern)
    
    Returns:
        X: Feature matrix (200 samples, 2 features)
        y: Target labels (binary: +1, -1)
    """
    np.random.seed(42)
    
    # Create XOR-like pattern
    n_samples = 50
    
    # Class 1: top-left and bottom-right quadrants
    class1_x1 = np.concatenate([np.random.normal(-2, 0.5, n_samples),
                                np.random.normal(2, 0.5, n_samples)])
    class1_x2 = np.concatenate([np.random.normal(2, 0.5, n_samples),
                                np.random.normal(-2, 0.5, n_samples)])
    
    # Class 2: top-right and bottom-left quadrants
    class2_x1 = np.concatenate([np.random.normal(2, 0.5, n_samples),
                                np.random.normal(-2, 0.5, n_samples)])
    class2_x2 = np.concatenate([np.random.normal(2, 0.5, n_samples),
                                np.random.normal(-2, 0.5, n_samples)])
    
    X = np.vstack([np.column_stack([class1_x1, class1_x2]),
                   np.column_stack([class2_x1, class2_x2])])
    y = np.hstack([np.ones(2*n_samples), -np.ones(2*n_samples)])
    
    return X, y

def compare_kernels():
    """
    Compare different kernel functions on non-linearly separable data
    Shows how different kernels handle the same data
    """
    print("=== KERNEL COMPARISON DEMONSTRATION ===")
    
    # Create non-linear data
    X_nonlinear, y_nonlinear = create_non_linear_data()
    
    # Define kernels to compare
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_names = ['Linear', 'Polynomial', 'RBF (Gaussian)', 'Sigmoid']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        # Train SVM with current kernel
        if kernel == 'poly':
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', degree=3, random_state=42)
        elif kernel == 'rbf':
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
        elif kernel == 'sigmoid':
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
        else:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)
        
        svm.fit(X_nonlinear, y_nonlinear)
        
        # Plot decision boundary
        x_min, x_max = X_nonlinear[:, 0].min() - 1, X_nonlinear[:, 0].max() + 1
        y_min, y_max = X_nonlinear[:, 1].min() - 1, X_nonlinear[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[i].contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.8)
        
        # Plot data points
        axes[i].scatter(X_nonlinear[y_nonlinear == 1][:, 0], X_nonlinear[y_nonlinear == 1][:, 1], 
                       c='blue', label='Class 1', s=30, alpha=0.7)
        axes[i].scatter(X_nonlinear[y_nonlinear == -1][:, 0], X_nonlinear[y_nonlinear == -1][:, 1], 
                       c='red', label='Class -1', s=30, alpha=0.7)
        
        # Plot support vectors
        support_vectors = svm.support_vectors_
        axes[i].scatter(support_vectors[:, 0], support_vectors[:, 1], 
                       c='green', s=100, alpha=0.8, linewidth=1, 
                       edgecolor='black', label='Support Vectors')
        
        axes[i].set_title(f'{name} Kernel (Accuracy: {svm.score(X_nonlinear, y_nonlinear):.3f})')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison results
    print("\nKernel Performance Comparison:")
    print("-" * 40)
    for kernel, name in zip(kernels, kernel_names):
        if kernel == 'poly':
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', degree=3, random_state=42)
        elif kernel == 'rbf':
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
        elif kernel == 'sigmoid':
            svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
        else:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)
        
        svm.fit(X_nonlinear, y_nonlinear)
        accuracy = svm.score(X_nonlinear, y_nonlinear)
        n_support = len(svm.support_vectors_)
        print(f"{name:15} | Accuracy: {accuracy:.3f} | Support Vectors: {n_support}")

# =============================================================================
# 4. SVM PARAMETER TUNING AND REGULARIZATION
# =============================================================================

"""
SVM PARAMETER TUNING
====================

Key Parameters:
1. C (Regularization): Controls trade-off between margin and misclassification
   - Low C: Large margin, more misclassifications (underfitting)
   - High C: Small margin, fewer misclassifications (overfitting)

2. γ (Gamma): Kernel coefficient for RBF, polynomial, and sigmoid kernels
   - Low γ: Smooth decision boundary, may underfit
   - High γ: Complex decision boundary, may overfit

3. Kernel-specific parameters:
   - Polynomial degree
   - RBF bandwidth
   - Sigmoid parameters

Best Practices:
- Use GridSearchCV for parameter tuning
- Apply cross-validation to avoid overfitting
- Scale features before training (SVM is sensitive to feature scales)
"""

def demonstrate_parameter_effects():
    """
    Show how different C and gamma values affect SVM performance
    Demonstrates overfitting vs underfitting
    """
    print("=== PARAMETER EFFECTS DEMONSTRATION ===")
    
    # Create complex non-linear data
    X_complex, y_complex = create_non_linear_data()
    
    # Add some noise to make it more challenging
    np.random.seed(42)
    noise_indices = np.random.choice(len(X_complex), size=20, replace=False)
    y_complex[noise_indices] *= -1  # Flip some labels
    
    # Define parameter combinations
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.1, 1, 10, 100]
    
    # Create subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            # Train SVM
            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            svm.fit(X_complex, y_complex)
            
            # Plot decision boundary
            x_min, x_max = X_complex[:, 0].min() - 1, X_complex[:, 0].max() + 1
            y_min, y_max = X_complex[:, 1].min() - 1, X_complex[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[i, j].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            axes[i, j].contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.8)
            
            # Plot data points
            axes[i, j].scatter(X_complex[y_complex == 1][:, 0], X_complex[y_complex == 1][:, 1], 
                              c='blue', s=20, alpha=0.7)
            axes[i, j].scatter(X_complex[y_complex == -1][:, 0], X_complex[y_complex == -1][:, 1], 
                              c='red', s=20, alpha=0.7)
            
            # Plot support vectors
            support_vectors = svm.support_vectors_
            axes[i, j].scatter(support_vectors[:, 0], support_vectors[:, 1], 
                              c='green', s=50, alpha=0.8, linewidth=1, 
                              edgecolor='black')
            
            accuracy = svm.score(X_complex, y_complex)
            n_support = len(support_vectors)
            axes[i, j].set_title(f'C={C}, γ={gamma}\nAcc: {accuracy:.3f}, SV: {n_support}')
            axes[i, j].set_xlabel('Feature 1')
            axes[i, j].set_ylabel('Feature 2')
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nParameter Effects Summary:")
    print("-" * 40)
    print("Low C (0.1): Large margin, may underfit")
    print("High C (100): Small margin, may overfit")
    print("Low γ (0.1): Smooth boundary, may underfit")
    print("High γ (100): Complex boundary, may overfit")

# =============================================================================
# 5. PRACTICAL IMPLEMENTATION WITH REAL DATASETS
# =============================================================================

"""
REAL-WORLD DATASET APPLICATIONS
===============================

We'll use the following datasets:
1. Iris Dataset: Multi-class classification (3 classes)
2. Breast Cancer Dataset: Binary classification
3. Wine Dataset: Multi-class classification (3 classes)

Implementation Steps:
1. Load and explore data
2. Preprocess data (scaling, encoding)
3. Split into training and testing sets
4. Train SVM model
5. Evaluate performance
6. Visualize results
"""

def iris_dataset_example():
    """
    Multi-class classification using the famous Iris dataset
    Demonstrates SVM for more than 2 classes
    """
    print("=== IRIS DATASET EXAMPLE ===")
    
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # For simplicity, use only 2 features for visualization
    X_2d = X[:, [0, 2]]  # Sepal length and Petal length
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm_iris = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_iris.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = svm_iris.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Iris Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Visualize decision boundaries
    plt.figure(figsize=(12, 8))
    
    # Create mesh grid
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = svm_iris.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.8)
    
    # Plot training data
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        idx = y_train == i
        plt.scatter(X_train_scaled[idx, 0], X_train_scaled[idx, 1], 
                   c=color, label=target_names[i], s=50, alpha=0.7)
    
    # Plot support vectors
    support_vectors = svm_iris.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               c='black', s=100, alpha=0.8, linewidth=2, 
               edgecolor='white', label='Support Vectors')
    
    plt.xlabel('Sepal Length (scaled)')
    plt.ylabel('Petal Length (scaled)')
    plt.title('SVM Classification - Iris Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return svm_iris, scaler

def breast_cancer_example():
    """
    Binary classification using the Breast Cancer Wisconsin dataset
    Demonstrates SVM for medical diagnosis
    """
    print("=== BREAST CANCER DATASET EXAMPLE ===")
    
    # Load Breast Cancer dataset
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Target classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM with RBF kernel
    svm_cancer = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm_cancer.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = svm_cancer.predict(X_test_scaled)
    y_pred_proba = svm_cancer.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Breast Cancer Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Breast Cancer Dataset')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Feature importance using coefficients (for linear kernel)
    # Note: For RBF kernel, we can't directly interpret feature importance
    # Let's try with a linear kernel to show feature importance
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    
    # Get feature importance from linear SVM
    feature_importance = np.abs(svm_linear.coef_[0])
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
    plt.yticks(range(len(top_features_idx)), 
               [feature_names[i] for i in top_features_idx])
    plt.xlabel('Feature Importance (|Coefficient|)')
    plt.title('Top 10 Most Important Features (Linear SVM)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return svm_cancer, scaler

def wine_dataset_example():
    """
    Multi-class classification using the Wine dataset
    Demonstrates SVM for wine classification
    """
    print("=== WINE DATASET EXAMPLE ===")
    
    # Load Wine dataset
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM with polynomial kernel
    svm_wine = SVC(kernel='poly', C=1.0, gamma='scale', degree=3, random_state=42)
    svm_wine.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = svm_wine.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Wine Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Visualize in 2D using PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Retrain SVM on PCA-transformed data
    svm_wine_pca = SVC(kernel='poly', C=1.0, gamma='scale', degree=3, random_state=42)
    svm_wine_pca.fit(X_train_pca, y_train)
    
    # Plot decision boundaries
    plt.figure(figsize=(12, 8))
    
    # Create mesh grid
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = svm_wine_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.8)
    
    # Plot training data
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        idx = y_train == i
        plt.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], 
                   c=color, label=target_names[i], s=50, alpha=0.7)
    
    # Plot support vectors
    support_vectors = svm_wine_pca.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               c='black', s=100, alpha=0.8, linewidth=2, 
               edgecolor='white', label='Support Vectors')
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('SVM Classification - Wine Dataset (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return svm_wine, scaler

# =============================================================================
# 6. SVM FOR REGRESSION
# =============================================================================

"""
SVM FOR REGRESSION (SVR)
========================

Support Vector Regression (SVR) is an extension of SVM for regression problems.

Key Concepts:
1. ε-tube: A region around the regression line where errors are tolerated
2. Support vectors: Points outside the ε-tube that influence the model
3. Kernel functions: Same as classification (linear, RBF, polynomial)

Parameters:
- C: Regularization parameter
- ε (epsilon): Width of the ε-tube
- γ (gamma): Kernel coefficient
"""

def demonstrate_svr():
    """
    Demonstrate Support Vector Regression with synthetic data
    Shows how SVR handles different types of relationships
    """
    print("=== SUPPORT VECTOR REGRESSION DEMONSTRATION ===")
    
    # Create synthetic regression data
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Define different SVR models
    svr_models = {
        'Linear SVR': SVR(kernel='linear', C=1.0, epsilon=0.1),
        'RBF SVR': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
        'Polynomial SVR': SVR(kernel='poly', C=1.0, gamma='scale', degree=3, epsilon=0.1)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in svr_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'mse': mse, 'r2': r2, 'y_pred': y_pred}
        
        print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        
        # Plot training data
        plt.scatter(X_train, y_train, c='blue', label='Training Data', s=20, alpha=0.7)
        
        # Plot test data
        plt.scatter(X_test, y_test, c='red', label='Test Data', s=20, alpha=0.7)
        
        # Plot predictions
        plt.scatter(X_test, result['y_pred'], c='green', label='Predictions', s=30, alpha=0.8)
        
        # Plot regression line
        X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
        y_plot = result['model'].predict(X_plot)
        plt.plot(X_plot, y_plot, 'k-', linewidth=2, label='SVR Prediction')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'{name}\nMSE: {result["mse"]:.4f}, R²: {result["r2"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. HYPERPARAMETER TUNING WITH GRID SEARCH
# =============================================================================

"""
HYPERPARAMETER TUNING
=====================

Grid Search with Cross-Validation:
- Systematically search through parameter combinations
- Use cross-validation to avoid overfitting
- Find optimal parameters for best generalization

Common Parameters to Tune:
- C: [0.1, 1, 10, 100]
- γ (gamma): [0.001, 0.01, 0.1, 1, 10]
- Kernel: ['linear', 'rbf', 'poly']
- Degree (for polynomial): [2, 3, 4]
"""

def demonstrate_hyperparameter_tuning():
    """
    Demonstrate hyperparameter tuning using GridSearchCV
    Shows how to find optimal parameters automatically
    """
    print("=== HYPERPARAMETER TUNING DEMONSTRATION ===")
    
    # Load a dataset for tuning
    X, y = datasets.load_breast_cancer(return_X_y=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    # Create base SVM model
    base_svm = SVC(random_state=42)
    
    # Perform grid search
    print("Performing Grid Search...")
    grid_search = GridSearchCV(
        base_svm, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Results
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Best model
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy with best model: {test_accuracy:.4f}")
    
    # Cross-validation scores for all parameter combinations
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Plot results for RBF kernel (most common)
    rbf_results = cv_results[cv_results['param_kernel'] == 'rbf']
    
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    pivot_table = rbf_results.pivot_table(
        values='mean_test_score', 
        index='param_gamma', 
        columns='param_C'
    )
    
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Grid Search Results - RBF Kernel')
    plt.xlabel('C (Regularization)')
    plt.ylabel('γ (Gamma)')
    plt.show()
    
    # Compare with default parameters
    default_svm = SVC(random_state=42)
    default_svm.fit(X_train_scaled, y_train)
    default_accuracy = default_svm.score(X_test_scaled, y_test)
    
    print(f"\nComparison:")
    print(f"Default parameters accuracy: {default_accuracy:.4f}")
    print(f"Tuned parameters accuracy: {test_accuracy:.4f}")
    print(f"Improvement: {test_accuracy - default_accuracy:.4f}")
    
    return best_svm, grid_search

# =============================================================================
# 8. ADVANTAGES AND DISADVANTAGES OF SVM
# =============================================================================

"""
ADVANTAGES AND DISADVANTAGES OF SVM
===================================

ADVANTAGES:
-----------
1. Effective in high-dimensional spaces
2. Memory efficient (uses support vectors only)
3. Versatile (different kernels for different data types)
4. Robust to overfitting in high-dimensional spaces
5. Can handle non-linear relationships through kernel tricks
6. Provides good generalization with proper parameter tuning

DISADVANTAGES:
--------------
1. Sensitive to feature scaling
2. Requires careful parameter tuning
3. Can be computationally expensive for large datasets
4. Black-box model (difficult to interpret)
5. Memory usage can be high for large datasets
6. May not work well with noisy data

WHEN TO USE SVM:
----------------
✅ High-dimensional data (text, images, genomics)
✅ Binary classification problems
✅ When you have limited training data
✅ When you need a robust, generalizable model
✅ When you can afford computational resources for tuning

WHEN NOT TO USE SVM:
--------------------
❌ Very large datasets (consider linear SVM or other algorithms)
❌ When interpretability is crucial
❌ When you need fast training/prediction
❌ When you have limited computational resources
❌ Multi-class problems with many classes (consider one-vs-one or one-vs-rest)
"""

# =============================================================================
# 9. MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function to run all demonstrations
    """
    print("SUPPORT VECTOR MACHINES - COMPREHENSIVE GUIDE")
    print("=" * 50)
    
    # 1. Linear SVM demonstration
    print("\n1. LINEAR SVM DEMONSTRATION")
    print("-" * 30)
    demonstrate_linear_svm()
    
    # 2. Kernel comparison
    print("\n2. KERNEL COMPARISON")
    print("-" * 30)
    compare_kernels()
    
    # 3. Parameter effects
    print("\n3. PARAMETER EFFECTS")
    print("-" * 30)
    demonstrate_parameter_effects()
    
    # 4. Real-world examples
    print("\n4. REAL-WORLD DATASET EXAMPLES")
    print("-" * 30)
    
    # Iris dataset
    print("\n4a. Iris Dataset (Multi-class)")
    iris_model, iris_scaler = iris_dataset_example()
    
    # Breast cancer dataset
    print("\n4b. Breast Cancer Dataset (Binary)")
    cancer_model, cancer_scaler = breast_cancer_example()
    
    # Wine dataset
    print("\n4c. Wine Dataset (Multi-class)")
    wine_model, wine_scaler = wine_dataset_example()
    
    # 5. SVR demonstration
    print("\n5. SUPPORT VECTOR REGRESSION")
    print("-" * 30)
    demonstrate_svr()
    
    # 6. Hyperparameter tuning
    print("\n6. HYPERPARAMETER TUNING")
    print("-" * 30)
    best_model, grid_search = demonstrate_hyperparameter_tuning()
    
    print("\n" + "=" * 50)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return {
        'iris_model': iris_model,
        'cancer_model': cancer_model,
        'wine_model': wine_model,
        'best_tuned_model': best_model,
        'grid_search': grid_search
    }

# =============================================================================
# 10. RUN THE DEMONSTRATIONS
# =============================================================================

if __name__ == "__main__":
    # Run all demonstrations
    models = main()
    
    print("\nFinal Summary:")
    print(f"- Iris SVM Model: {type(models['iris_model']).__name__}")
    print(f"- Cancer SVM Model: {type(models['cancer_model']).__name__}")
    print(f"- Wine SVM Model: {type(models['wine_model']).__name__}")
    print(f"- Best Tuned Model: {type(models['best_tuned_model']).__name__}")
    
    print("\nYou can now use these trained models for predictions!")
    print("Example usage:")
    print("  predictions = models['iris_model'].predict(new_data)")
    print("  probabilities = models['cancer_model'].predict_proba(new_data)")