#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predictive Models for Fuel Transaction Anomaly Detection

This module implements and evaluates various machine learning models for classifying
potentially anomalous fuel transactions based on multiple heuristic flags.
The implemented models include:
1. Linear Support Vector Machine
2. Naive Bayes
3. XGBoost
4. Random Forest
5. Multi-layer Perceptron (Neural Network)
6. Stochastic Gradient Descent

Author: Jared Tavares
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib import cm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a custom transformer to convert sparse matrix to dense
class DenseTransformer(TransformerMixin):
    """
    Custom transformer to convert sparse matrices to dense format.
    Required for certain models like Naive Bayes that don't work with sparse input.
    """
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, 'toarray') else X

def load_prepare_data(file_path='data/Final Transactions With Flags.csv'):
    """
    Load and prepare the data for classification.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
        
    Returns:
    --------
    tuple
        X, y, categorical_features, numerical_features
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load data with flags
        data = pd.read_csv(file_path, low_memory=False)
        
        # Log the shape of the loaded data
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Make sure class labels are consistent (change any 3 to 2)
        if 3 in data['Number_of_Flags'].values:
            logger.info("Converting Number_of_Flags=3 to Number_of_Flags=2")
            data['Number_of_Flags'] = data['Number_of_Flags'].replace(3, 2)
        
        # Add a column for fuel price
        data['Fuel Price'] = data.apply(
            lambda row: row['Coastal Diesel'] if row['Fuel Type'] == 'Diesel' else row['Coastal Petrol'], 
            axis=1
        )
        
        # Preprocess district feature
        process_categorical_feature(data, 'District', 'District_new', min_count=10000)
        
        # Preprocess vehicle make feature
        process_categorical_feature(data, 'VEHICLE MAKE', 'Make_new', min_count=10000)
        
        # Preprocess rate card category feature
        process_categorical_feature(data, 'RATE CARD CATEGORY', 'Cat_new', min_count=5000)
        
        # Set the "MANAGED MAINTENANCE" value in Cat_new to "Other"
        data['Cat_new'] = data['Cat_new'].replace('MANAGED MAINTENANCE', 'Other')
        
        # Trim the Cat_new column to remove leading and trailing spaces
        data['Cat_new'] = data['Cat_new'].str.strip()
        
        # Define features and target
        categorical_features = ['District_new', 'Make_new', 'Fuel Type', 'Cat_new']
        numerical_features = ['Transaction Amount', 'No. of Litres', 'Fuel Price']
        
        X = data.drop(['Number_of_Flags'], axis=1)
        y = data['Number_of_Flags']
        
        logger.info(f"Data preparation completed with {len(categorical_features)} categorical features and {len(numerical_features)} numerical features")
        
        return X, y, categorical_features, numerical_features
    
    except Exception as e:
        logger.error(f"Error loading or preparing data: {str(e)}")
        raise

def process_categorical_feature(df, original_column, new_column, min_count=10000):
    """
    Process a categorical feature by grouping low-frequency categories into 'Other'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame
    original_column : str
        Original categorical column name
    new_column : str
        New column name for processed data
    min_count : int
        Minimum count threshold for keeping a category
    """
    # Get the names of the categories with counts below threshold
    categories = df[original_column].value_counts()
    low_count_categories = categories[categories < min_count].index
    
    # Replace the low-count categories with 'Other'
    df[new_column] = df[original_column].replace(low_count_categories, 'Other')
    
    logger.info(f"Processed {original_column} into {new_column}, replaced {len(low_count_categories)} categories with 'Other'")

def train_linear_svm(X_train, y_train, X_test, y_test, categorical_features, numerical_features):
    """
    Train and evaluate a Linear Support Vector Machine model.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series
        Testing data
    categorical_features, numerical_features : list
        Lists of categorical and numerical feature names
        
    Returns:
    --------
    tuple
        Trained model, predictions, pipeline
    """
    logger.info("Training Linear SVM model")
    
    # Preprocess the data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Apply feature selection
    selector = SelectKBest(score_func=f_classif, k=15)
    
    # Define the pipeline
    pipeline_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('classifier', LinearSVC(random_state=1, class_weight='balanced', dual='auto'))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.1, 0.5, 1, 5],
        'classifier__class_weight': ['balanced']
    }
    
    # Use a smaller subset for hyperparameter tuning if dataset is large
    if X_train.shape[0] > 50000:
        logger.info("Using subset for hyperparameter tuning due to large dataset")
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train, y_train, test_size=0.9, random_state=1
        )
    else:
        X_train_subset, y_train_subset = X_train, y_train
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        pipeline_svm, param_grid, cv=StratifiedKFold(n_splits=3), 
        scoring='accuracy', n_jobs=-1
    )
    
    # Fit the model with GridSearchCV
    logger.info("Performing grid search for Linear SVM")
    grid_search.fit(X_train_subset, y_train_subset)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    logger.info(f"Linear SVM Best Parameters: {grid_search.best_params_}")
    logger.info(f"Linear SVM Test Accuracy: {grid_search.score(X_test, y_test):.4f}")
    
    return best_model, y_pred, grid_search

def train_naive_bayes(X_train, y_train, X_test, y_test, categorical_features, numerical_features):
    """
    Train and evaluate a Naive Bayes model.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series
        Testing data
    categorical_features, numerical_features : list
        Lists of categorical and numerical feature names
        
    Returns:
    --------
    tuple
        Trained model, predictions, pipeline
    """
    logger.info("Training Naive Bayes model")
    
    # Preprocessors for numerical and categorical features
    numeric_transformer_nb = Pipeline(steps=[
        ('scaler_nb', StandardScaler())
    ])
    
    categorical_transformer_nb = Pipeline(steps=[
        ('onehot_nb', OneHotEncoder(handle_unknown='ignore')),
        ('to_dense_nb', DenseTransformer())
    ])
    
    # Combine preprocessors
    preprocessor_nb = ColumnTransformer(
        transformers=[
            ('num_nb', numeric_transformer_nb, numerical_features),
            ('cat_nb', categorical_transformer_nb, categorical_features)
        ])
    
    # Create a pipeline for Naive Bayes
    pipeline_nb = make_pipeline(
        preprocessor_nb,
        GaussianNB()
    )
    
    # Fit the model
    logger.info("Fitting Naive Bayes model")
    pipeline_nb.fit(X_train, y_train)
    
    # Make predictions
    y_pred_nb = pipeline_nb.predict(X_test)
    
    logger.info(f"Naive Bayes Test Accuracy: {pipeline_nb.score(X_test, y_test):.4f}")
    
    return pipeline_nb, y_pred_nb, pipeline_nb

def train_xgboost(X_train, y_train, X_test, y_test, categorical_features, numerical_features):
    """
    Train and evaluate an XGBoost model.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series
        Testing data
    categorical_features, numerical_features : list
        Lists of categorical and numerical feature names
        
    Returns:
    --------
    tuple
        Trained model, predictions, pipeline
    """
    logger.info("Training XGBoost model")
    
    # Define transformations for categorical and numerical features
    preprocessor_xgb = ColumnTransformer(
        transformers=[
            ('num_xgb', StandardScaler(), numerical_features),
            ('cat_xgb', OneHotEncoder(), categorical_features)
        ])
    
    # Create a pipeline
    pipeline_xgb = Pipeline([
        ('preprocessor_xgb', preprocessor_xgb),
        ('classifier_xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    # Optional: Define parameters for GridSearchCV
    param_grid_xgb = {
        'classifier_xgb__n_estimators': [100, 200],
        'classifier_xgb__learning_rate': [0.01, 0.1],
    }
    
    # Optional: Create GridSearchCV object
    grid_search_xgb = GridSearchCV(
        pipeline_xgb, param_grid_xgb, cv=3, verbose=1, n_jobs=-1
    )
    
    # Fit the model
    logger.info("Fitting XGBoost model")
    pipeline_xgb.fit(X_train, y_train)
    
    # Make predictions
    y_pred_xgb = pipeline_xgb.predict(X_test)
    
    logger.info(f"XGBoost Test Accuracy: {pipeline_xgb.score(X_test, y_test):.4f}")
    
    return pipeline_xgb, y_pred_xgb, pipeline_xgb

def train_random_forest(X_train, y_train, X_test, y_test, categorical_features, numerical_features):
    """
    Train and evaluate a Random Forest model.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series
        Testing data
    categorical_features, numerical_features : list
        Lists of categorical and numerical feature names
        
    Returns:
    --------
    tuple
        Trained model, predictions, pipeline
    """
    logger.info("Training Random Forest model")
    
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Create classifier
    rf_classifier = RandomForestClassifier(random_state=1, class_weight='balanced')
    
    # Define parameter grid
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [15, 20],
        'classifier__min_samples_split': [5, 10]
    }
    
    # Create pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])
    
    # Create grid search
    rf_grid_search = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=StratifiedKFold(n_splits=3), 
        scoring='accuracy', n_jobs=-1
    )
    
    # Fit the model
    logger.info("Performing grid search for Random Forest")
    rf_grid_search.fit(X_train, y_train)
    
    # Get best model
    rf_best_model = rf_grid_search.best_estimator_
    
    # Make predictions
    rf_y_pred = rf_best_model.predict(X_test)
    
    logger.info(f"Random Forest Best Parameters: {rf_grid_search.best_params_}")
    logger.info(f"Random Forest Test Accuracy: {rf_grid_search.score(X_test, y_test):.4f}")
    
    return rf_best_model, rf_y_pred, rf_grid_search

def train_mlp(X_train, y_train, X_test, y_test, categorical_features, numerical_features):
    """
    Train and evaluate a Multi-layer Perceptron (Neural Network) model.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series
        Testing data
    categorical_features, numerical_features : list
        Lists of categorical and numerical feature names
        
    Returns:
    --------
    tuple
        Trained model, predictions, pipeline
    """
    logger.info("Training Multi-layer Perceptron model")
    
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Create classifier
    mlp_classifier = MLPClassifier(random_state=1)
    
    # Define parameter grid
    mlp_param_grid = {
        'classifier__hidden_layer_sizes': [(100,50), (100,50,20)],
        'classifier__alpha': [0.001, 0.01],
        'classifier__learning_rate_init': [0.001]
    }
    
    # Create pipeline
    mlp_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', mlp_classifier)
    ])
    
    # Create grid search
    mlp_grid_search = GridSearchCV(
        mlp_pipeline, mlp_param_grid, cv=StratifiedKFold(n_splits=3), 
        scoring='accuracy', n_jobs=-1
    )
    
    # Fit the model
    logger.info("Performing grid search for Multi-layer Perceptron")
    mlp_grid_search.fit(X_train, y_train)
    
    # Get best model
    mlp_best_model = mlp_grid_search.best_estimator_
    
    # Make predictions
    mlp_y_pred = mlp_best_model.predict(X_test)
    
    logger.info(f"MLP Best Parameters: {mlp_grid_search.best_params_}")
    logger.info(f"MLP Test Accuracy: {mlp_grid_search.score(X_test, y_test):.4f}")
    
    return mlp_best_model, mlp_y_pred, mlp_grid_search

def train_sgd(X_train, y_train, X_test, y_test, categorical_features, numerical_features):
    """
    Train and evaluate a Stochastic Gradient Descent model.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series
        Testing data
    categorical_features, numerical_features : list
        Lists of categorical and numerical feature names
        
    Returns:
    --------
    tuple
        Trained model, predictions, pipeline
    """
    logger.info("Training SGD Classifier model")
    
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Create classifier
    sgd_classifier = SGDClassifier(random_state=1, class_weight='balanced')
    
    # Define parameter grid
    sgd_param_grid = {
        'classifier__loss': ['hinge', 'log_loss', 'modified_huber'],
        'classifier__alpha': [0.0001, 0.001],
        'classifier__penalty': ['l1', 'l2', 'elasticnet']
    }
    
    # Create pipeline
    sgd_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', sgd_classifier)
    ])
    
    # Create grid search
    sgd_grid_search = GridSearchCV(
        sgd_pipeline, sgd_param_grid, cv=StratifiedKFold(n_splits=3), 
        scoring='accuracy', n_jobs=-1
    )
    
    # Fit the model
    logger.info("Performing grid search for SGD Classifier")
    sgd_grid_search.fit(X_train, y_train)
    
    # Get best model
    sgd_best_model = sgd_grid_search.best_estimator_
    
    # Make predictions
    sgd_y_pred = sgd_best_model.predict(X_test)
    
    logger.info(f"SGD Best Parameters: {sgd_grid_search.best_params_}")
    logger.info(f"SGD Test Accuracy: {sgd_grid_search.score(X_test, y_test):.4f}")
    
    return sgd_best_model, sgd_y_pred, sgd_grid_search

def plot_confusion_matrix(y_true, y_pred, model_name, dpi=300):
    """
    Plot and save a confusion matrix.
    
    Parameters:
    -----------
    y_true : pandas.Series
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    model_name : str
        Name of the model
    dpi : int
        Resolution for saved plot
    """
    logger.info(f"Plotting confusion matrix for {model_name}")
    matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate the total number of samples for each true class
    true_class_totals = matrix.sum(axis=1)
    
    # Calculate the percentage of true class for each cell in the confusion matrix
    matrix_percent = matrix / true_class_totals[:, np.newaxis] * 100
    
    # Create a text annotation matrix for displaying both values and percentages
    annot_matrix = [[f"{value}\n({percent:.2f}%)" for value, percent in zip(row, row_percent)]
                    for row, row_percent in zip(matrix, matrix_percent)]
    
    # Create output directory if it doesn't exist
    os.makedirs('plots/modelling', exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, annot=annot_matrix, fmt='', cmap='cividis', square=True, cbar=False, annot_kws={"size": 12})
    plt.xlabel('Predicted Number of Flags', size=14)
    plt.ylabel('True Number of Flags', size=14)
    
    # Set the font size of the ticks on the x and y axis
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot as a PDF file
    plt.tight_layout()
    plt.savefig(f'plots/modelling/{model_name}_confusion_matrix.pdf', format='pdf', dpi=dpi)
    plt.close()

def plot_roc_curve_multiclass(model, X_test, y_test, model_name):
    """
    Plot and save ROC curves for a multi-class classifier.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test labels
    model_name : str
        Name of the model
    """
    logger.info(f"Plotting ROC curves for {model_name}")
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    n_classes = y_test_binarized.shape[1]
    
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X_test)
    else:
        # Use decision_function for models that don't have predict_proba
        try:
            y_pred = model.decision_function(X_test)
            # Reshape if needed
            if len(y_pred.shape) == 1:
                y_pred = np.column_stack([1 - y_pred, y_pred])
        except:
            logger.warning(f"{model_name} does not support predict_proba or decision_function. ROC curve cannot be plotted.")
            return
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Create output directory if it doesn't exist
    os.makedirs('plots/modelling', exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    cividis_colors = cm.cividis(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), cividis_colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, 
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=13)
    plt.tight_layout()
    plt.savefig(f'plots/modelling/{model_name}_roc_curve.pdf', format='pdf', dpi=300)
    plt.close()

def save_model(model, filename):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    filename : str
        Filename for saved model
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model to {filename}: {str(e)}")

def load_model(filename):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Filename of saved model
        
    Returns:
    --------
    sklearn estimator
        Loaded model
    """
    try:
        model = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filename}: {str(e)}")
        return None

def main():
    """Main function to execute model training, evaluation, and visualization."""
    # Create output directories
    os.makedirs('plots/modelling', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    logger.info("Starting model training and evaluation")
    X, y, categorical_features, numerical_features = load_prepare_data()
    
    # Split data using stratified sampling
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    logger.info(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
    
    # Train all models
    model_results = {}
    
    # Linear SVM
    try:
        svm_model, svm_preds, svm_pipeline = train_linear_svm(
            X_train, y_train, X_test, y_test, categorical_features, numerical_features
        )
        plot_confusion_matrix(y_test, svm_preds, 'Linear SVM')
        plot_roc_curve_multiclass(svm_model, X_test, y_test, 'Linear SVM')
        save_model(svm_model, 'models/linear_svm_model.pkl')
        model_results['Linear SVM'] = (svm_model, svm_preds)
    except Exception as e:
        logger.error(f"Error training Linear SVM: {str(e)}")
    
    # Naive Bayes
    try:
        nb_model, nb_preds, nb_pipeline = train_naive_bayes(
            X_train, y_train, X_test, y_test, categorical_features, numerical_features
        )
        plot_confusion_matrix(y_test, nb_preds, 'Naive Bayes')
        plot_roc_curve_multiclass(nb_model, X_test, y_test, 'Naive Bayes')
        save_model(nb_model, 'models/naive_bayes_model.pkl')
        model_results['Naive Bayes'] = (nb_model, nb_preds)
    except Exception as e:
        logger.error(f"Error training Naive Bayes: {str(e)}")
    
    # XGBoost
    try:
        xgb_model, xgb_preds, xgb_pipeline = train_xgboost(
            X_train, y_train, X_test, y_test, categorical_features, numerical_features
        )
        plot_confusion_matrix(y_test, xgb_preds, 'XGBoost')
        plot_roc_curve_multiclass(xgb_model, X_test, y_test, 'XGBoost')
        save_model(xgb_model, 'models/xgboost_model.pkl')
        model_results['XGBoost'] = (xgb_model, xgb_preds)
    except Exception as e:
        logger.error(f"Error training XGBoost: {str(e)}")
    
    # Random Forest
    try:
        rf_model, rf_preds, rf_pipeline = train_random_forest(
            X_train, y_train, X_test, y_test, categorical_features, numerical_features
        )
        plot_confusion_matrix(y_test, rf_preds, 'Random Forest')
        plot_roc_curve_multiclass(rf_model, X_test, y_test, 'Random Forest')
        save_model(rf_model, 'models/random_forest_model.pkl')
        model_results['Random Forest'] = (rf_model, rf_preds)
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
    
    # Multi-layer Perceptron
    try:
        mlp_model, mlp_preds, mlp_pipeline = train_mlp(
            X_train, y_train, X_test, y_test, categorical_features, numerical_features
        )
        plot_confusion_matrix(y_test, mlp_preds, 'Multi-layer Perceptron')
        plot_roc_curve_multiclass(mlp_model, X_test, y_test, 'Multi-layer Perceptron')
        save_model(mlp_model, 'models/mlp_model.pkl')
        model_results['Multi-layer Perceptron'] = (mlp_model, mlp_preds)
    except Exception as e:
        logger.error(f"Error training Multi-layer Perceptron: {str(e)}")
    
    # SGD Classifier
    try:
        sgd_model, sgd_preds, sgd_pipeline = train_sgd(
            X_train, y_train, X_test, y_test, categorical_features, numerical_features
        )
        plot_confusion_matrix(y_test, sgd_preds, 'SGD Classifier')
        plot_roc_curve_multiclass(sgd_model, X_test, y_test, 'SGD Classifier')
        save_model(sgd_model, 'models/sgd_model.pkl')
        model_results['SGD Classifier'] = (sgd_model, sgd_preds)
    except Exception as e:
        logger.error(f"Error training SGD Classifier: {str(e)}")
    
    # Summarize results
    results_summary = {}
    for model_name, (model, preds) in model_results.items():
        report = classification_report(y_test, preds, output_dict=True)
        results_summary[model_name] = {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'class_0_f1': report['0']['f1-score'],
            'class_1_f1': report['1']['f1-score'],
            'class_2_f1': report['2']['f1-score']
        }
    
    # Convert summary to DataFrame and save
    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv('models/model_performance_summary.csv')
    
    logger.info(f"Model training and evaluation completed. Results saved to models/model_performance_summary.csv")
    
    return model_results, summary_df

if __name__ == "__main__":
    # Execute the main function if the script is run directly
    model_results, summary_df = main()
    
    # Print the performance summary
    print("\nModel Performance Summary:")
    print(summary_df)