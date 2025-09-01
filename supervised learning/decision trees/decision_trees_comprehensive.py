"""
Decision Trees: Comprehensive Theoretical Guide and Implementation
================================================================

This file provides a detailed theoretical overview of decision trees, practical examples,
and comparisons with other machine learning algorithms for classification and regression tasks.

Author: Learning ML
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionTreesComprehensive:
    """
    Comprehensive class covering decision trees theory and implementation
    """
    
    def __init__(self):
        self.classification_results = {}
        self.regression_results = {}
        
    def theoretical_overview(self):
        """
        Detailed theoretical overview of decision trees
        """
        print("=" * 80)
        print("DECISION TREES: THEORETICAL OVERVIEW")
        print("=" * 80)
        
        print("\n1. WHAT ARE DECISION TREES?")
        print("-" * 40)
        print("""
        Decision trees are hierarchical, tree-like structures used for:
        • Classification: Predicting categorical outcomes
        • Regression: Predicting continuous values
        
        Structure:
        • Root Node: Starting point with entire dataset
        • Internal Nodes: Decision points with splitting criteria
        • Leaf Nodes: Final predictions/outcomes
        • Branches: Paths connecting nodes based on conditions
        """)
        
        print("\n2. HOW DECISION TREES WORK")
        print("-" * 40)
        print("""
        Algorithm Steps:
        1. Start with root node containing all data
        2. Find best feature and threshold to split data
        3. Create child nodes based on split
        4. Repeat recursively until stopping criteria met
        5. Assign predictions to leaf nodes
        
        Key Concepts:
        • Splitting Criteria: How to choose best split
        • Impurity Measures: Quantify node homogeneity
        • Pruning: Prevent overfitting
        • Tree Depth: Control complexity
        """)
        
        print("\n3. IMPURITY MEASURES")
        print("-" * 40)
        print("""
        Classification Impurity Measures:
        
        a) Gini Impurity:
           Gini = 1 - Σ(p_i²) where p_i is probability of class i
           • Range: [0, 1-1/k] where k is number of classes
           • 0 = pure node (all samples same class)
           • Higher values = more impure
        
        b) Entropy:
           Entropy = -Σ(p_i * log2(p_i))
           • Range: [0, log2(k)]
           • 0 = pure node
           • log2(k) = maximum impurity
        
        c) Information Gain:
           IG = Parent_Impurity - Σ(|S_v|/|S| * Child_v_Impurity)
           • Measures reduction in impurity after split
           • Higher IG = better split
        
        Regression Impurity Measures:
        
        a) Mean Squared Error (MSE):
           MSE = (1/n) * Σ(y_i - y_mean)²
           • Measures variance within node
        
        b) Mean Absolute Error (MAE):
           MAE = (1/n) * Σ|y_i - y_median|
           • Less sensitive to outliers
        """)
        
        print("\n4. SPLITTING CRITERIA")
        print("-" * 40)
        print("""
        Best Split Selection:
        
        1. For each feature:
           - Sort unique values
           - Consider each value as potential threshold
           - Calculate impurity for resulting split
        
        2. Choose split that maximizes:
           - Information Gain (classification)
           - Variance reduction (regression)
        
        3. Stopping Criteria:
           - Maximum tree depth reached
           - Minimum samples per node
           - Minimum impurity decrease
           - All samples in node belong to same class
        """)
        
        print("\n5. ADVANTAGES OF DECISION TREES")
        print("-" * 40)
        print("""
        ✓ Interpretable and easy to understand
        ✓ Can handle both numerical and categorical data
        ✓ No assumptions about data distribution
        ✓ Can capture non-linear relationships
        ✓ Feature importance ranking
        ✓ Robust to outliers
        ✓ Fast training and prediction
        ✓ Can handle missing values
        """)
        
        print("\n6. DISADVANTAGES OF DECISION TREES")
        print("-" * 40)
        print("""
        ✗ Prone to overfitting (high variance)
        ✗ Unstable (small changes in data → different trees)
        ✗ Can create biased trees if classes are imbalanced
        ✗ May not generalize well to unseen data
        ✗ Limited to axis-parallel splits
        ✗ Can be computationally expensive for large datasets
        """)
        
        print("\n7. PRUNING TECHNIQUES")
        print("-" * 40)
        print("""
        Pre-pruning (Early Stopping):
        • max_depth: Maximum tree depth
        • min_samples_split: Minimum samples to split node
        • min_samples_leaf: Minimum samples in leaf node
        • min_impurity_decrease: Minimum impurity reduction
        
        Post-pruning (Cost Complexity Pruning):
        • Remove subtrees that don't improve performance
        • Balance tree complexity vs. accuracy
        • Use validation set to determine optimal pruning
        """)
    
    def classification_example(self):
        """
        Comprehensive classification example using Iris dataset
        """
        print("\n" + "=" * 80)
        print("CLASSIFICATION EXAMPLE: IRIS DATASET")
        print("=" * 80)
        
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        print(f"\nDataset Info:")
        print(f"Features: {feature_names}")
        print(f"Classes: {target_names}")
        print(f"Shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train decision tree
        dt_classifier = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        dt_classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = dt_classifier.predict(X_test)
        y_pred_proba = dt_classifier.predict_proba(X_test)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"CV Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = dt_classifier.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df)
        
        # Visualize tree
        plt.figure(figsize=(20, 10))
        plot_tree(dt_classifier, 
                 feature_names=feature_names,
                 class_names=target_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
        plt.show()
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names,
                   yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Store results for comparison
        self.classification_results['Decision Tree'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return dt_classifier
    
    def regression_example(self):
        """
        Comprehensive regression example using Diabetes dataset
        """
        print("\n" + "=" * 80)
        print("REGRESSION EXAMPLE: DIABETES DATASET")
        print("=" * 80)
        
        # Load dataset
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        feature_names = diabetes.feature_names
        
        print(f"\nDataset Info:")
        print(f"Features: {feature_names}")
        print(f"Shape: {X.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Target mean: {y.mean():.2f}")
        print(f"Target std: {y.std():.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train decision tree
        dt_regressor = DecisionTreeRegressor(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        dt_regressor.fit(X_train, y_train)
        
        # Predictions
        y_pred = dt_regressor.predict(X_test)
        
        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(dt_regressor, X, y, cv=5, scoring='r2')
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"CV Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = dt_regressor.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df)
        
        # Visualize tree
        plt.figure(figsize=(20, 10))
        plot_tree(dt_regressor, 
                 feature_names=feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title('Decision Tree Regressor Visualization', fontsize=16, fontweight='bold')
        plt.show()
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Residuals plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Store results for comparison
        self.regression_results['Decision Tree'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return dt_regressor
    
    def hyperparameter_tuning(self):
        """
        Demonstrate hyperparameter tuning for decision trees
        """
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING")
        print("=" * 80)
        
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        # Grid search
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Results summary
        results_df = pd.DataFrame(grid_search.cv_results_)
        best_results = results_df[results_df['rank_test_score'] == 1]
        
        print(f"\nBest Model Performance:")
        print(f"Mean CV Score: {best_results['mean_test_score'].iloc[0]:.4f}")
        print(f"Std CV Score: {best_results['std_test_score'].iloc[0]:.4f}")
        
        return grid_search.best_estimator_
    
    def algorithm_comparison(self):
        """
        Compare decision trees with other algorithms
        """
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON")
        print("=" * 80)
        
        # Classification comparison
        print("\n1. CLASSIFICATION COMPARISON")
        print("-" * 40)
        
        # Load classification data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features for algorithms that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            if name in ['Logistic Regression', 'SVM', 'KNN']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            self.classification_results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name:20} | Accuracy: {accuracy:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Regression comparison
        print("\n2. REGRESSION COMPARISON")
        print("-" * 40)
        
        # Load regression data
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        reg_models = {
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Train and evaluate models
        for name, model in reg_models.items():
            if name in ['Linear Regression', 'SVR', 'KNN']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            self.regression_results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"{name:20} | R²: {r2:.4f} | RMSE: {rmse:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Visualize comparison
        self._plot_comparison()
    
    def _plot_comparison(self):
        """
        Plot comparison results
        """
        # Classification comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification accuracy
        names = list(self.classification_results.keys())
        accuracies = [self.classification_results[name]['accuracy'] for name in names]
        cv_means = [self.classification_results[name]['cv_mean'] for name in names]
        cv_stds = [self.classification_results[name]['cv_std'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        ax1.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8, yerr=cv_stds, capsize=5)
        
        ax1.set_xlabel('Algorithms')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regression R² comparison
        reg_names = list(self.regression_results.keys())
        r2_scores = [self.regression_results[name]['r2'] for name in reg_names]
        reg_cv_means = [self.regression_results[name]['cv_mean'] for name in reg_names]
        reg_cv_stds = [self.regression_results[name]['cv_std'] for name in reg_names]
        
        x = np.arange(len(reg_names))
        
        ax2.bar(x - width/2, r2_scores, width, label='Test R²', alpha=0.8)
        ax2.bar(x + width/2, reg_cv_means, width, label='CV Mean', alpha=0.8, yerr=reg_cv_stds, capsize=5)
        
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Regression Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(reg_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def when_decision_trees_perform_better(self):
        """
        Detailed analysis of when decision trees perform better
        """
        print("\n" + "=" * 80)
        print("WHEN DECISION TREES PERFORM BETTER")
        print("=" * 80)
        
        print("\n1. SCENARIOS WHERE DECISION TREES EXCEL:")
        print("-" * 50)
        print("""
        ✓ Non-linear relationships:
          - When data has complex, non-linear patterns
          - Other algorithms (linear models) struggle with these
        
        ✓ Mixed data types:
          - Numerical and categorical features together
          - No need for feature scaling or encoding
        
        ✓ Interpretability requirements:
          - When you need to explain decisions
          - Business stakeholders need transparency
        
        ✓ Feature interactions:
          - When features interact in complex ways
          - Trees can capture these interactions naturally
        
        ✓ Outlier robustness:
          - Less sensitive to outliers than linear models
          - Splits based on percentiles, not means
        
        ✓ Missing values:
          - Can handle missing values naturally
          - No need for imputation in many cases
        
        ✓ Small to medium datasets:
          - When dataset size is manageable
          - Can capture complex patterns without overfitting
        """)
        
        print("\n2. SCENARIOS WHERE OTHER ALGORITHMS ARE BETTER:")
        print("-" * 50)
        print("""
        ✗ Large datasets with linear relationships:
          - Linear models are faster and more efficient
          - Trees can become computationally expensive
        
        ✗ High-dimensional data:
          - Trees may overfit with many features
          - Regularization methods (L1/L2) work better
        
        ✗ When stability is crucial:
          - Small data changes can drastically alter tree structure
          - Ensemble methods (Random Forest) are more stable
        
        ✗ When you need probability estimates:
          - Trees provide discrete predictions
          - Logistic regression provides well-calibrated probabilities
        
        ✗ When data is highly correlated:
          - Trees may favor one feature over correlated ones
          - Linear models handle correlations better
        """)
        
        print("\n3. PRACTICAL GUIDELINES:")
        print("-" * 50)
        print("""
        Start with Decision Trees when:
        • You need interpretable results
        • Data has mixed types (numerical + categorical)
        • You suspect non-linear relationships
        • Dataset is small to medium size
        • You want to understand feature importance
        
        Consider other algorithms when:
        • Dataset is very large (>100K samples)
        • Features are highly correlated
        • You need probability estimates
        • Performance is critical and data is linear
        • You need stable, reproducible results
        
        Use ensemble methods (Random Forest) when:
        • You want better performance than single trees
        • Stability is important
        • You still need interpretability
        • Dealing with high-dimensional data
        """)
        
        print("\n4. HYBRID APPROACHES:")
        print("-" * 50)
        print("""
        Best of both worlds:
        
        • Random Forest:
          - Multiple trees reduce overfitting
          - Better generalization than single trees
          - Still interpretable through feature importance
        
        • Gradient Boosting:
          - Sequential learning from tree errors
          - Often best performance
          - Less interpretable than single trees
        
        • XGBoost/LightGBM:
          - Optimized implementations
          - Handle missing values automatically
          - Built-in regularization
        
        • Stacking:
          - Combine multiple algorithms
          - Use meta-learner to combine predictions
          - Can leverage strengths of different approaches
        """)

def main():
    """
    Main function to run comprehensive decision tree analysis
    """
    print("DECISION TREES COMPREHENSIVE GUIDE")
    print("=" * 80)
    
    # Initialize the comprehensive guide
    dt_guide = DecisionTreesComprehensive()
    
    # Run theoretical overview
    dt_guide.theoretical_overview()
    
    # Run practical examples
    dt_guide.classification_example()
    dt_guide.regression_example()
    
    # Run hyperparameter tuning
    best_model = dt_guide.hyperparameter_tuning()
    
    # Run algorithm comparison
    dt_guide.algorithm_comparison()
    
    # Run when decision trees perform better analysis
    dt_guide.when_decision_trees_perform_better()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE GUIDE COMPLETED!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• Decision trees are excellent for interpretable, non-linear modeling")
    print("• They handle mixed data types without preprocessing")
    print("• Use ensemble methods for better performance and stability")
    print("• Consider the trade-off between interpretability and performance")
    print("• Always validate with cross-validation and test sets")

if __name__ == "__main__":
    main()
