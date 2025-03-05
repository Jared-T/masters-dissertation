#!/usr/bin/env python
# coding: utf-8

# # Classification Model Validation and Robustness
# 
# This script extends the original classification model implementation with comprehensive
# validation techniques to evaluate model robustness and performance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as XGBClassifier
import os
from matplotlib.ticker import FixedLocator, FixedFormatter
from tqdm import tqdm
import joblib
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
sns.set_palette('cividis')

def load_data():
    """
    Load and prepare the data for classification.
    
    Returns:
    --------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Testing labels
    feature_names : list
        Names of the features
    """
    # Load data with flags
    data = pd.read_csv(os.path.join("data", "Final Transactions With Flags.csv"))
    
    # Make sure class labels are consistent
    data['Number_of_Flags'] = data['Number_of_Flags'].replace(3, 2)
    
    # Define features and target
    categorical_features = ['District_new', 'Make_new', 'Fuel Type', 'Cat_new']
    numerical_features = ['Transaction Amount', 'No. of Litres', 'Fuel Price']
    
    X = data.drop(['Number_of_Flags'], axis=1)
    y = data['Number_of_Flags']
    
    # Create a list of feature names for later use in feature importance
    feature_names = numerical_features + [f"{col}_{cat}" for col in categorical_features 
                                        for cat in X[col].unique()]
    
    # Split data into training and testing sets using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, categorical_features, numerical_features

def create_preprocessing_pipeline(categorical_features, numerical_features):
    """
    Create a preprocessing pipeline for the data.
    
    Parameters:
    -----------
    categorical_features : list
        List of categorical feature names
    numerical_features : list
        List of numerical feature names
    
    Returns:
    --------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    """
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    return preprocessor

def evaluate_model_stability(model, preprocessor, X_train, y_train, n_runs=10, test_size=0.2, save_dir='plots/model_stability'):
    """
    Evaluate the stability of a model across multiple train-test splits.
    
    Parameters:
    -----------
    model : estimator
        The classification model to evaluate
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    n_runs : int
        Number of different random train-test splits to evaluate
    test_size : float
        Proportion of data to use for testing in each split
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    stability_results : dict
        Dictionary containing stability metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize results storage
    accuracy_scores = []
    f1_macro_scores = []
    class_f1_scores = {0: [], 1: [], 2: []}
    
    # Run multiple train-test splits
    for i in tqdm(range(n_runs), desc="Stability Evaluation"):
        # Split the training data again for this run
        X_run_train, X_run_test, y_run_train, y_run_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=i, stratify=y_train
        )
        
        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit the pipeline
        pipeline.fit(X_run_train, y_run_train)
        
        # Make predictions
        y_run_pred = pipeline.predict(X_run_test)
        
        # Get classification report
        report = classification_report(y_run_test, y_run_pred, output_dict=True)
        
        # Store metrics
        accuracy_scores.append(report['accuracy'])
        f1_macro_scores.append(report['macro avg']['f1-score'])
        
        # Store class-specific F1 scores
        for cls in [0, 1, 2]:
            class_f1_scores[cls].append(report[str(cls)]['f1-score'])
    
    # Calculate stability metrics
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    f1_macro_mean = np.mean(f1_macro_scores)
    f1_macro_std = np.std(f1_macro_scores)
    
    class_f1_means = {cls: np.mean(scores) for cls, scores in class_f1_scores.items()}
    class_f1_stds = {cls: np.std(scores) for cls, scores in class_f1_scores.items()}
    
    # Plot stability metrics
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy distribution
    plt.hist(accuracy_scores, bins=10, alpha=0.7, color='blue', label='Accuracy')
    plt.axvline(accuracy_mean, color='blue', linestyle='--', 
               label=f'Mean Accuracy: {accuracy_mean:.3f} ± {accuracy_std:.3f}')
    
    # Plot F1 macro distribution
    plt.hist(f1_macro_scores, bins=10, alpha=0.7, color='green', label='F1 Macro')
    plt.axvline(f1_macro_mean, color='green', linestyle='--', 
               label=f'Mean F1 Macro: {f1_macro_mean:.3f} ± {f1_macro_std:.3f}')
    
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Stability of Model Metrics Across Different Train-Test Splits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_stability.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot class-specific F1 scores
    plt.figure(figsize=(10, 6))
    
    for cls, scores in class_f1_scores.items():
        plt.hist(scores, bins=10, alpha=0.5, label=f'Class {cls} F1')
        plt.axvline(class_f1_means[cls], color=plt.gca().get_lines()[-1].get_color(), linestyle='--', 
                   label=f'Class {cls} Mean: {class_f1_means[cls]:.3f} ± {class_f1_stds[cls]:.3f}')
    
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.title('Stability of Class-Specific F1 Scores Across Different Train-Test Splits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_f1_stability.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Return stability metrics
    stability_results = {
        'accuracy_mean': accuracy_mean,
        'accuracy_std': accuracy_std,
        'f1_macro_mean': f1_macro_mean,
        'f1_macro_std': f1_macro_std,
        'class_f1_means': class_f1_means,
        'class_f1_stds': class_f1_stds
    }
    
    return stability_results

def k_fold_cross_validation(model, preprocessor, X_train, y_train, k=5, save_dir='plots/cross_validation'):
    """
    Perform k-fold cross-validation on the model.
    
    Parameters:
    -----------
    model : estimator
        The classification model to evaluate
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    k : int
        Number of folds for cross-validation
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    cv_results : dict
        Dictionary containing cross-validation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Create a stratified k-fold cross-validator
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    
    # Initialize results storage
    fold_accuracy = []
    fold_f1_macro = []
    fold_reports = []
    fold_matrices = []
    
    # Perform cross-validation
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        # Split data for this fold
        X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
        
        # Fit the pipeline
        pipeline.fit(X_fold_train, y_fold_train)
        
        # Make predictions
        y_fold_pred = pipeline.predict(X_fold_test)
        
        # Get metrics
        report = classification_report(y_fold_test, y_fold_pred, output_dict=True)
        matrix = confusion_matrix(y_fold_test, y_fold_pred)
        
        # Store metrics
        fold_accuracy.append(report['accuracy'])
        fold_f1_macro.append(report['macro avg']['f1-score'])
        fold_reports.append(report)
        fold_matrices.append(matrix)
    
    # Calculate average metrics
    avg_accuracy = np.mean(fold_accuracy)
    avg_f1_macro = np.mean(fold_f1_macro)
    
    # Extract class-specific metrics across folds
    class_metrics = {}
    for cls in [0, 1, 2]:
        cls_str = str(cls)
        class_metrics[cls] = {
            'precision': [report[cls_str]['precision'] for report in fold_reports],
            'recall': [report[cls_str]['recall'] for report in fold_reports],
            'f1-score': [report[cls_str]['f1-score'] for report in fold_reports]
        }
    
    # Plot average metrics
    plt.figure(figsize=(10, 6))
    
    # Create bar positions
    bar_width = 0.35
    r1 = np.arange(k)
    r2 = [x + bar_width for x in r1]
    
    # Plot bars
    plt.bar(r1, fold_accuracy, width=bar_width, label='Accuracy', color='blue', alpha=0.7)
    plt.bar(r2, fold_f1_macro, width=bar_width, label='F1 Macro', color='green', alpha=0.7)
    
    # Add fold labels
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'{k}-Fold Cross-Validation Results')
    plt.xticks([r + bar_width/2 for r in range(k)], [f'Fold {i+1}' for i in range(k)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_validation_metrics.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot class-specific metrics
    plt.figure(figsize=(15, 5))
    
    # Create subplots for precision, recall, and F1
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot precision
    for cls in [0, 1, 2]:
        ax1.plot(range(1, k+1), class_metrics[cls]['precision'], marker='o', label=f'Class {cls}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision by Class and Fold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot recall
    for cls in [0, 1, 2]:
        ax2.plot(range(1, k+1), class_metrics[cls]['recall'], marker='o', label=f'Class {cls}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall by Class and Fold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot F1
    for cls in [0, 1, 2]:
        ax3.plot(range(1, k+1), class_metrics[cls]['f1-score'], marker='o', label=f'Class {cls}')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score by Class and Fold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_validation_class_metrics.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Calculate and plot average confusion matrix
    avg_matrix = np.mean(fold_matrices, axis=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_matrix, annot=True, fmt='.1f', cmap='cividis', 
               xticklabels=['No Flags', 'One Flag', 'Multiple Flags'],
               yticklabels=['No Flags', 'One Flag', 'Multiple Flags'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Average Confusion Matrix Across Folds')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_confusion_matrix.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Return cross-validation results
    cv_results = {
        'fold_accuracy': fold_accuracy,
        'fold_f1_macro': fold_f1_macro,
        'avg_accuracy': avg_accuracy,
        'avg_f1_macro': avg_f1_macro,
        'class_metrics': class_metrics,
        'avg_matrix': avg_matrix
    }
    
    return cv_results

def plot_learning_curve(model, preprocessor, X_train, y_train, save_dir='plots/learning_curve'):
    """
    Generate and plot a learning curve for the model.
    
    Parameters:
    -----------
    model : estimator
        The classification model to evaluate
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    learning_curve_results : dict
        Dictionary containing learning curve data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Define training set sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train, train_sizes=train_sizes, 
        cv=5, scoring='f1_macro', n_jobs=-1
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                    alpha=0.1, color='green')
    
    plt.xlabel('Training Examples')
    plt.ylabel('F1 Macro Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Return learning curve data
    learning_curve_results = {
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std
    }
    
    return learning_curve_results

def evaluate_model_on_test(model, preprocessor, X_train, X_test, y_train, y_test, 
                           feature_names, save_dir='plots/test_evaluation'):
    """
    Evaluate the model on the test set and generate comprehensive performance metrics.
    
    Parameters:
    -----------
    model : estimator
        The classification model to evaluate
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training labels
    y_test : pandas.Series
        Testing labels
    feature_names : list
        Names of the features
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    test_results : dict
        Dictionary containing test evaluation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Get prediction probabilities (if the model supports it)
    if hasattr(pipeline, 'predict_proba'):
        y_pred_proba = pipeline.predict_proba(X_test)
    else:
        y_pred_proba = None
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='cividis', 
               xticklabels=['No Flags', 'One Flag', 'Multiple Flags'],
               yticklabels=['No Flags', 'One Flag', 'Multiple Flags'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot ROC curve and Precision-Recall curve (if prediction probabilities are available)
    if y_pred_proba is not None:
        # Convert target to one-hot encoding for ROC curve
        y_test_onehot = pd.get_dummies(y_test).values
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        
        for i in range(3):  # For each class
            fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'), format='pdf', dpi=300)
        plt.close()
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        
        for i in range(3):  # For each class
            precision, recall, _ = precision_recall_curve(y_test_onehot[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_test_onehot[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.pdf'), format='pdf', dpi=300)
        plt.close()
        
        # Plot calibration curve
        plt.figure(figsize=(10, 8))
        
        for i in range(3):  # For each class
            prob_true, prob_pred = calibration_curve(y_test_onehot[:, i], y_pred_proba[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Class {i}')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_curve.pdf'), format='pdf', dpi=300)
        plt.close()
    
    # Calculate and plot feature importance (if the model supports it)
    if hasattr(model, 'feature_importances_'):
        # Extract feature importance directly from the model
        feature_importance = model.feature_importances_
        
        # Sort feature importances
        sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        # Get the feature names from the preprocessor
        transformed_features = preprocessor.transform(X_test.iloc[:5])
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names_out = preprocessor.get_feature_names_out()
        else:
            # If get_feature_names_out is not available, use a simpler approach
            feature_names_out = [f'Feature {i}' for i in range(transformed_features.shape[1])]
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names_out[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.pdf'), format='pdf', dpi=300)
        plt.close()
    else:
        # Use permutation importance as an alternative
        perm_importance = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=1)
        
        # Sort permutation importances
        sorted_idx = perm_importance.importances_mean.argsort()[-20:]  # Top 20 features
        
        # Get the feature names from the preprocessor
        transformed_features = preprocessor.transform(X_test.iloc[:5])
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names_out = preprocessor.get_feature_names_out()
        else:
            # If get_feature_names_out is not available, use a simpler approach
            feature_names_out = [f'Feature {i}' for i in range(transformed_features.shape[1])]
        
        # Plot permutation importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names_out[i] for i in sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Top 20 Feature Importances (Permutation)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'permutation_importance.pdf'), format='pdf', dpi=300)
        plt.close()
    
    # Return test evaluation results
    test_results = {
        'report': report,
        'matrix': matrix,
        'accuracy': report['accuracy'],
        'f1_macro': report['macro avg']['f1-score'],
        'class_f1': {cls: report[str(cls)]['f1-score'] for cls in [0, 1, 2]}
    }
    
    return test_results

def hyperparameter_optimization(model_type, preprocessor, X_train, y_train, save_dir='plots/hyperparameter_tuning'):
    """
    Perform hyperparameter optimization for the selected model type.
    
    Parameters:
    -----------
    model_type : str
        Type of the model ('mlp', 'rf', or 'xgb')
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    best_model : estimator
        The best model from hyperparameter optimization
    opt_results : dict
        Dictionary containing optimization results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define parameter distributions for each model type
    if model_type == 'mlp':
        model = MLPClassifier(random_state=1)
        param_distributions = {
            'classifier__hidden_layer_sizes': [(100,), (100, 50), (100, 50, 20)],
            'classifier__alpha': uniform(0.0001, 0.01),
            'classifier__learning_rate_init': uniform(0.001, 0.01),
            'classifier__max_iter': randint(100, 500),
            'classifier__early_stopping': [True, False]
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=1)
        param_distributions = {
            'classifier__n_estimators': randint(50, 300),
            'classifier__max_depth': randint(10, 50),
            'classifier__min_samples_split': randint(2, 20),
            'classifier__min_samples_leaf': randint(1, 10),
            'classifier__class_weight': ['balanced', 'balanced_subsample', None]
        }
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=1)
        param_distributions = {
            'classifier__n_estimators': randint(50, 300),
            'classifier__max_depth': randint(3, 10),
            'classifier__learning_rate': uniform(0.01, 0.3),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4),
            'classifier__gamma': uniform(0, 1)
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Create a randomized search CV
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_distributions, 
        n_iter=50, cv=5, scoring='f1_macro', 
        n_jobs=-1, random_state=1, verbose=1
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_.named_steps['classifier']
    
    # Plot the top parameters
    results = pd.DataFrame(random_search.cv_results_)
    
    # Get the parameter names
    param_names = [key.replace('classifier__', '') for key in param_distributions.keys()]
    
    # Plot the influence of each parameter on the f1 macro score
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(param_names):
        param_key = f'param_classifier__{param}'
        
        # Check if the parameter exists in the results
        if param_key in results.columns:
            # Extract parameter values and scores
            param_values = results[param_key].values
            mean_scores = results['mean_test_score'].values
            
            # Create subplot
            plt.subplot(2, 3, i+1)
            
            # Check if parameter values are numeric
            try:
                param_values = param_values.astype(float)
                plt.scatter(param_values, mean_scores, alpha=0.7)
                
                # Add trendline for numeric parameters
                if len(np.unique(param_values)) > 5:  # Only add trendline if there are enough unique values
                    z = np.polyfit(param_values, mean_scores, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(param_values), p(sorted(param_values)), 'r--')
            except (ValueError, TypeError):
                # For categorical parameters, use boxplot
                param_df = pd.DataFrame({'param': param_values, 'score': mean_scores})
                sns.boxplot(x='param', y='score', data=param_df)
            
            plt.title(f'Effect of {param}')
            plt.xlabel(param)
            plt.ylabel('F1 Macro Score')
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_parameter_influence.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot the distribution of scores
    plt.figure(figsize=(10, 6))
    plt.hist(results['mean_test_score'], bins=20, alpha=0.7, color='blue')
    plt.axvline(random_search.best_score_, color='red', linestyle='--', 
               label=f'Best Score: {random_search.best_score_:.3f}')
    plt.xlabel('F1 Macro Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cross-Validation Scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_score_distribution.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Save the best parameters
    best_params = random_search.best_params_
    
    # Return the best model and optimization results
    opt_results = {
        'best_params': best_params,
        'best_score': random_search.best_score_,
        'cv_results': results.to_dict()
    }
    
    return best_model, opt_results

def test_model_robustness(model, preprocessor, X_train, X_test, y_train, y_test, save_dir='plots/robustness'):
    """
    Test the robustness of the model to various perturbations.
    
    Parameters:
    -----------
    model : estimator
        The classification model to evaluate
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training labels
    y_test : pandas.Series
        Testing labels
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    robustness_results : dict
        Dictionary containing robustness metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Get baseline performance
    y_baseline_pred = pipeline.predict(X_test)
    baseline_report = classification_report(y_test, y_baseline_pred, output_dict=True)
    baseline_accuracy = baseline_report['accuracy']
    baseline_f1_macro = baseline_report['macro avg']['f1-score']
    
    # Define perturbation types
    perturbation_results = []
    
    # 1. Missing values perturbation
    for missing_pct in [0.05, 0.1, 0.2]:
        # Create a copy of the test set
        X_perturbed = X_test.copy()
        
        # Randomly set values to NaN
        mask = np.random.random(X_perturbed.shape) < missing_pct
        X_perturbed = X_perturbed.mask(mask)
        
        # Make predictions
        y_perturbed_pred = pipeline.predict(X_perturbed)
        perturbed_report = classification_report(y_test, y_perturbed_pred, output_dict=True)
        
        # Store results
        perturbation_results.append({
            'perturbation_type': 'Missing Values',
            'perturbation_level': f'{missing_pct*100}%',
            'accuracy': perturbed_report['accuracy'],
            'f1_macro': perturbed_report['macro avg']['f1-score'],
            'accuracy_change': perturbed_report['accuracy'] - baseline_accuracy,
            'f1_macro_change': perturbed_report['macro avg']['f1-score'] - baseline_f1_macro
        })
    
    # 2. Noise perturbation for numerical features
    for noise_level in [0.05, 0.1, 0.2]:
        # Create a copy of the test set
        X_perturbed = X_test.copy()
        
        # Add random noise to numerical features
        for col in X_perturbed.select_dtypes(include=['float64', 'int64']).columns:
            col_std = X_perturbed[col].std()
            noise = np.random.normal(0, noise_level * col_std, size=X_perturbed.shape[0])
            X_perturbed[col] = X_perturbed[col] + noise
        
        # Make predictions
        y_perturbed_pred = pipeline.predict(X_perturbed)
        perturbed_report = classification_report(y_test, y_perturbed_pred, output_dict=True)
        
        # Store results
        perturbation_results.append({
            'perturbation_type': 'Numerical Noise',
            'perturbation_level': f'{noise_level*100}%',
            'accuracy': perturbed_report['accuracy'],
            'f1_macro': perturbed_report['macro avg']['f1-score'],
            'accuracy_change': perturbed_report['accuracy'] - baseline_accuracy,
            'f1_macro_change': perturbed_report['macro avg']['f1-score'] - baseline_f1_macro
        })
    
    # 3. Categorical swap perturbation
    for swap_pct in [0.05, 0.1, 0.2]:
        # Create a copy of the test set
        X_perturbed = X_test.copy()
        
        # Randomly swap categorical values
        for col in X_perturbed.select_dtypes(include=['object', 'category']).columns:
            mask = np.random.random(X_perturbed.shape[0]) < swap_pct
            unique_values = X_perturbed[col].unique()
            
            for idx in X_perturbed[mask].index:
                current_value = X_perturbed.loc[idx, col]
                other_values = [v for v in unique_values if v != current_value]
                
                if other_values:
                    X_perturbed.loc[idx, col] = np.random.choice(other_values)
        
        # Make predictions
        y_perturbed_pred = pipeline.predict(X_perturbed)
        perturbed_report = classification_report(y_test, y_perturbed_pred, output_dict=True)
        
        # Store results
        perturbation_results.append({
            'perturbation_type': 'Categorical Swap',
            'perturbation_level': f'{swap_pct*100}%',
            'accuracy': perturbed_report['accuracy'],
            'f1_macro': perturbed_report['macro avg']['f1-score'],
            'accuracy_change': perturbed_report['accuracy'] - baseline_accuracy,
            'f1_macro_change': perturbed_report['macro avg']['f1-score'] - baseline_f1_macro
        })
    
    # Convert to DataFrame
    perturbation_df = pd.DataFrame(perturbation_results)
    
    # Plot accuracy changes
    plt.figure(figsize=(12, 6))
    
    # Group by perturbation type and level
    for ptype in perturbation_df['perturbation_type'].unique():
        ptype_data = perturbation_df[perturbation_df['perturbation_type'] == ptype]
        plt.plot(ptype_data['perturbation_level'], ptype_data['accuracy_change'], 
                marker='o', label=ptype)
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Perturbation Level')
    plt.ylabel('Change in Accuracy')
    plt.title('Model Robustness to Different Perturbations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_robustness.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot F1 macro changes
    plt.figure(figsize=(12, 6))
    
    # Group by perturbation type and level
    for ptype in perturbation_df['perturbation_type'].unique():
        ptype_data = perturbation_df[perturbation_df['perturbation_type'] == ptype]
        plt.plot(ptype_data['perturbation_level'], ptype_data['f1_macro_change'], 
                marker='o', label=ptype)
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Perturbation Level')
    plt.ylabel('Change in F1 Macro Score')
    plt.title('Model Robustness to Different Perturbations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_macro_robustness.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Return robustness results
    robustness_results = {
        'baseline_accuracy': baseline_accuracy,
        'baseline_f1_macro': baseline_f1_macro,
        'perturbation_results': perturbation_df.to_dict('records')
    }
    
    return robustness_results

def compare_models(models, preprocessor, X_train, X_test, y_train, y_test, save_dir='plots/model_comparison'):
    """
    Compare the performance of multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to compare (name: model)
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training labels
    y_test : pandas.Series
        Testing labels
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    comparison_results : dict
        Dictionary containing comparison metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize results storage
    model_results = []
    
    # Evaluate each model
    for name, model in models.items():
        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        model_results.append({
            'model_name': name,
            'accuracy': report['accuracy'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'class_0_f1': report['0']['f1-score'],
            'class_1_f1': report['1']['f1-score'],
            'class_2_f1': report['2']['f1-score']
        })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(model_results)
    
    # Plot overall metrics
    plt.figure(figsize=(10, 6))
    
    # Create bar positions
    bar_width = 0.2
    r1 = np.arange(len(comparison_df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Plot bars
    plt.bar(r1, comparison_df['accuracy'], width=bar_width, label='Accuracy', color='blue', alpha=0.7)
    plt.bar(r2, comparison_df['f1_macro'], width=bar_width, label='F1 Macro', color='green', alpha=0.7)
    plt.bar(r3, comparison_df['precision_macro'], width=bar_width, label='Precision Macro', color='red', alpha=0.7)
    plt.bar(r4, comparison_df['recall_macro'], width=bar_width, label='Recall Macro', color='purple', alpha=0.7)
    
    # Add model labels
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks([r + bar_width*1.5 for r in range(len(comparison_df))], comparison_df['model_name'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_overall.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot class-specific F1 scores
    plt.figure(figsize=(10, 6))
    
    # Create bar positions
    r1 = np.arange(len(comparison_df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Plot bars
    plt.bar(r1, comparison_df['class_0_f1'], width=bar_width, label='Class 0 F1', color='blue', alpha=0.7)
    plt.bar(r2, comparison_df['class_1_f1'], width=bar_width, label='Class 1 F1', color='green', alpha=0.7)
    plt.bar(r3, comparison_df['class_2_f1'], width=bar_width, label='Class 2 F1', color='red', alpha=0.7)
    
    # Add model labels
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison by Class')
    plt.xticks([r + bar_width for r in range(len(comparison_df))], comparison_df['model_name'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_by_class.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Return comparison results
    comparison_results = {
        'model_comparison': comparison_df.to_dict('records')
    }
    
    return comparison_results

# Run the functions if this script is executed directly
if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names, categorical_features, numerical_features = load_data()
    
    print("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    
    print("Running hyperparameter optimization for MLP...")
    mlp_best, mlp_opt_results = hyperparameter_optimization('mlp', preprocessor, X_train, y_train)
    
    print("Running hyperparameter optimization for Random Forest...")
    rf_best, rf_opt_results = hyperparameter_optimization('rf', preprocessor, X_train, y_train)
    
    print("Running hyperparameter optimization for XGBoost...")
    xgb_best, xgb_opt_results = hyperparameter_optimization('xgb', preprocessor, X_train, y_train)
    
    # Save the best models
    models = {
        'MLP': mlp_best,
        'Random Forest': rf_best,
        'XGBoost': xgb_best
    }
    
    print("Comparing models...")
    comparison_results = compare_models(models, preprocessor, X_train, X_test, y_train, y_test)
    
    # Select the best model (MLP based on previous findings)
    best_model = mlp_best
    
    print("Evaluating best model stability...")
    stability_results = evaluate_model_stability(best_model, preprocessor, X_train, y_train)
    
    print("Performing cross-validation...")
    cv_results = k_fold_cross_validation(best_model, preprocessor, X_train, y_train)
    
    print("Generating learning curve...")
    learning_curve_results = plot_learning_curve(best_model, preprocessor, X_train, y_train)
    
    print("Evaluating on test set...")
    test_results = evaluate_model_on_test(best_model, preprocessor, X_train, X_test, y_train, y_test, feature_names)
    
    print("Testing model robustness...")
    robustness_results = test_model_robustness(best_model, preprocessor, X_train, X_test, y_train, y_test)
    
    print("All analyses complete!")