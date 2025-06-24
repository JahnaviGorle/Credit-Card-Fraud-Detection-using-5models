import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st

def load_and_inspect_data(uploaded_file):
    """
    Load and perform initial inspection of the data
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded CSV file
        
    Returns:
    --------
    DataFrame, dict
        The loaded DataFrame and a dictionary with data statistics
    """
    if uploaded_file is None:
        return None, None
    
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None
    
    # Calculate statistics
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }
    
    if 'Class' in df.columns:
        stats['fraud_transactions'] = df['Class'].sum()
        stats['non_fraud_transactions'] = len(df) - df['Class'].sum()
        stats['fraud_percentage'] = (df['Class'].sum() / len(df)) * 100
    
    return df, stats

def preprocess_data(df, test_size=0.2, random_state=42, apply_smote=True, scaler_type='standard'):
    """
    Preprocess data for model training
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    test_size : float
        Test size for train-test split
    random_state : int
        Random state for reproducibility
    apply_smote : bool
        Whether to apply SMOTE for handling class imbalance
    scaler_type : str
        Type of scaler to use ('standard' or 'robust')
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    # Check if 'Class' column exists
    if 'Class' not in df.columns:
        st.error("Data must contain a 'Class' column for classification")
        return None, None, None, None, None
    
    # Separate features and target
    if 'Time' in df.columns:
        # Time is often not useful as a direct feature
        df = df.drop('Time', axis=1)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if requested
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        st.info("Applied SMOTE to balance the dataset")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def get_feature_names(df):
    """
    Get feature names from DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
        
    Returns:
    --------
    list
        List of feature names
    """
    if df is None:
        return []
    
    # Exclude Class and Time columns
    features = [col for col in df.columns if col not in ['Class', 'Time']]
    return features

def get_sample_fraud_and_normal(df, n_samples=5):
    """
    Get sample fraud and normal transactions for display
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    n_samples : int
        Number of samples to retrieve
        
    Returns:
    --------
    tuple
        Sample fraud and normal transactions
    """
    if df is None or 'Class' not in df.columns:
        return None, None
    
    fraud = df[df['Class'] == 1].sample(min(n_samples, sum(df['Class'])))
    normal = df[df['Class'] == 0].sample(min(n_samples, sum(df['Class'] == 0)))
    
    return fraud, normal
