import numpy as np
import streamlit as st
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def train_logistic_regression(X_train, y_train, max_iter=1000, C=1.0):
    """
    Train a Logistic Regression model
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    max_iter : int
        Maximum iterations
    C : float
        Regularization parameter
        
    Returns:
    --------
    LogisticRegression, float
        Trained model and training time
    """
    start_time = time.time()
    model = LogisticRegression(max_iter=max_iter, C=C, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time

def train_decision_tree(X_train, y_train, max_depth=10, min_samples_split=2):
    """
    Train a Decision Tree model
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    max_depth : int
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split
        
    Returns:
    --------
    DecisionTreeClassifier, float
        Trained model and training time
    """
    start_time = time.time()
    model = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """
    Train a Random Forest model
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum depth of trees
        
    Returns:
    --------
    RandomForestClassifier, float
        Trained model and training time
    """
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    n_estimators : int
        Number of trees
    learning_rate : float
        Learning rate
    max_depth : int
        Maximum depth of trees
        
    Returns:
    --------
    XGBClassifier, float
        Trained model and training time
    """
    start_time = time.time()
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time

def train_deep_learning(X_train, y_train, hidden_layer_sizes=(128, 64, 32), max_iter=200):
    """
    Train a Deep Learning model using MLPClassifier (sklearn)
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    hidden_layer_sizes : tuple
        Sizes of hidden layers
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    MLPClassifier, float
        Trained model and training time
    """
    start_time = time.time()
    
    # Display progress information
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Training Deep Learning Model: Starting...")
    
    # Create a more complex MLPClassifier as a substitute for the deep learning model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Update progress
    progress_bar.progress(1.0)
    status_text.text("Training Deep Learning Model: 100% Complete")
    
    train_time = time.time() - start_time
    progress_bar.empty()
    status_text.empty()
    
    return model, train_time

def evaluate_model(model, X_test, y_test, model_type='sklearn'):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
    model_type : str
        Type of model ('sklearn')
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report (as string)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

def get_feature_importance(model, feature_names, model_type):
    """
    Get feature importance from models
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    model_type : str
        Type of model
        
    Returns:
    --------
    dict or None
        Feature importance dictionary
    """
    try:
        if model_type == 'logistic_regression':
            importances = abs(model.coef_[0])
        elif model_type in ['decision_tree', 'random_forest']:
            importances = model.feature_importances_
        elif model_type == 'xgboost':
            importances = model.feature_importances_
        elif model_type in ['mlp', 'deep_learning']:
            # Neural networks don't provide direct feature importance
            return None
        else:
            return None
        
        # Create dictionary of feature importance
        importance_dict = dict(zip(feature_names, importances))
        
        return importance_dict
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return None
