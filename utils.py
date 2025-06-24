import pandas as pd
import numpy as np
import streamlit as st
import time
import base64
from io import BytesIO

def get_model_icon(model_name):
    """
    Get SVG icon for a model
    
    Parameters:
    -----------
    model_name : str
        Name of the model
        
    Returns:
    --------
    str
        SVG icon
    """
    icons = {
        'Logistic Regression': """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                   <rect width="24" height="24" rx="4" fill="rgba(255,75,75,0.8)"/>
                                   <path d="M4 16L9 11 M9 11L14 16 M14 16L20 10" stroke="white" stroke-width="2" stroke-linecap="round"/>
                                 </svg>""",
        
        'Decision Tree': """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect width="24" height="24" rx="4" fill="rgba(46,134,193,0.8)"/>
                            <circle cx="12" cy="6" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="7" cy="12" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="17" cy="12" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="4" cy="18" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="10" cy="18" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="14" cy="18" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="20" cy="18" r="2" stroke="white" stroke-width="1.5"/>
                            <path d="M11 7L8 10" stroke="white" stroke-width="1.5"/>
                            <path d="M13 7L16 10" stroke="white" stroke-width="1.5"/>
                            <path d="M6 14L5 16" stroke="white" stroke-width="1.5"/>
                            <path d="M8 14L9 16" stroke="white" stroke-width="1.5"/>
                            <path d="M16 14L15 16" stroke="white" stroke-width="1.5"/>
                            <path d="M18 14L19 16" stroke="white" stroke-width="1.5"/>
                          </svg>""",
                          
        'Random Forest': """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect width="24" height="24" rx="4" fill="rgba(39,174,96,0.8)"/>
                            <path d="M5 16V8L8 10L11 8V16" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M13 16V8L16 10L19 8V16" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M4 19H20" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M7 5L9 7L11 5" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M15 5L17 7L19 5" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                          </svg>""",
                          
        'XGBoost': """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <rect width="24" height="24" rx="4" fill="rgba(243,156,18,0.8)"/>
                      <path d="M4 16L8 12L12 16L16 12L20 8" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                      <path d="M4 8L8 12L12 8L16 12L20 16" stroke="white" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="2 2"/>
                    </svg>""",
                    
        'Deep Learning': """<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <rect width="24" height="24" rx="4" fill="rgba(142,68,173,0.8)"/>
                            <circle cx="6" cy="8" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="6" cy="12" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="6" cy="16" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="12" cy="6" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="12" cy="12" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="12" cy="18" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="18" cy="8" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="18" cy="12" r="2" stroke="white" stroke-width="1.5"/>
                            <circle cx="18" cy="16" r="2" stroke="white" stroke-width="1.5"/>
                            <path d="M8 8H10" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M8 12H10" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M8 16H10" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M14 8H16" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M14 12H16" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M14 16H16" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                          </svg>"""
    }
    
    return icons.get(model_name, "")

def generate_fraud_detection_explanation(model_name):
    """
    Generate explanation about how the model detects fraud
    
    Parameters:
    -----------
    model_name : str
        Name of the model
        
    Returns:
    --------
    str
        Explanation text
    """
    explanations = {
        'Logistic Regression': """
            <p>Logistic Regression detects fraud by calculating the probability of a transaction being fraudulent based on weighted input features. 
            It establishes a decision boundary in the feature space, classifying transactions above a threshold as fraudulent.</p>
            <p>The model is highly interpretable, with coefficients directly indicating each feature's influence on fraud detection. 
            Positive coefficients increase fraud probability, while negative ones decrease it.</p>
        """,
        
        'Decision Tree': """
            <p>Decision Trees detect fraud by creating a tree-like structure of decision rules based on transaction features.
            Starting at the root node, the algorithm applies sequential yes/no questions to classify transactions.</p>
            <p>This approach excels at capturing non-linear patterns and feature interactions that might indicate fraudulent behavior,
            and provides transparent decision paths that can be easily visualized and interpreted.</p>
        """,
        
        'Random Forest': """
            <p>Random Forest aggregates multiple decision trees to detect fraud with higher accuracy and less overfitting.
            Each tree votes on the transaction classification, with the majority vote determining the final prediction.</p>
            <p>The algorithm introduces randomness by sampling transactions with replacement (bagging) and selecting random feature
            subsets at each split, making it robust against outliers and noise while capturing complex fraud patterns.</p>
        """,
        
        'XGBoost': """
            <p>XGBoost (Extreme Gradient Boosting) sequentially builds multiple weak decision trees, with each new tree
            focusing on correcting the mistakes of previous trees.</p>
            <p>This powerful gradient boosting approach excels at fraud detection by capturing complex patterns and interactions between features,
            using regularization techniques to prevent overfitting, and handling imbalanced data effectively - critical for the rare nature of fraud cases.</p>
        """,
        
        'Deep Learning': """
            <p>Deep Neural Networks detect fraud by processing transaction data through multiple interconnected layers of neurons,
            automatically learning hierarchical feature representations.</p>
            <p>The network progressively extracts more abstract patterns across layers, enabling it to identify subtle fraud indicators
            that might be missed by other algorithms. This approach can capture complex non-linear relationships and feature interactions,
            making it effective for detecting sophisticated fraud patterns.</p>
        """
    }
    
    return explanations.get(model_name, "")

def generate_sample_data(n_samples=1000, fraud_ratio=0.1):
    """
    Generate sample credit card transaction data
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    fraud_ratio : float
        Ratio of fraudulent transactions
        
    Returns:
    --------
    DataFrame
        Sample transaction data
    """
    np.random.seed(42)
    
    # Generate feature names
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    # Generate data
    data = []
    
    # Generate legitimate transactions
    n_legitimate = int(n_samples * (1 - fraud_ratio))
    for _ in range(n_legitimate):
        time = np.random.randint(0, 172800)  # Time in seconds (48 hours)
        features = np.random.normal(0, 1, 28)  # V1-V28
        amount = np.random.lognormal(3, 1)  # Transaction amount
        transaction = [time] + list(features) + [amount, 0]  # Class 0 = legitimate
        data.append(transaction)
    
    # Generate fraudulent transactions
    n_fraud = int(n_samples * fraud_ratio)
    for _ in range(n_fraud):
        time = np.random.randint(0, 172800)
        features = np.random.normal(-0.5, 2, 28)  # Different distribution for fraud
        amount = np.random.lognormal(4, 2)  # Potentially higher amounts
        transaction = [time] + list(features) + [amount, 1]  # Class 1 = fraud
        data.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    
    return df

def explain_features():
    """
    Generate explanation about features in credit card fraud datasets
    
    Returns:
    --------
    str
        Explanation text
    """
    explanation = """
    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(5px);">
        <h3 style="margin-top: 0;">Credit Card Fraud Detection Features</h3>
        
        <p>In the typical credit card fraud detection dataset (like the popular Kaggle dataset), features include:</p>
        
        <ul>
            <li><strong>Time</strong>: Seconds elapsed between the first transaction and the current one</li>
            <li><strong>V1-V28</strong>: Principal components obtained with PCA transformation (anonymized for confidentiality)</li>
            <li><strong>Amount</strong>: Transaction amount</li>
            <li><strong>Class</strong>: Target variable (1 for fraud, 0 for legitimate)</li>
        </ul>
        
        <p>These features are typically the result of a PCA transformation to protect sensitive information. The original features would include:</p>
        
        <ul>
            <li>Transaction details (amount, time, location)</li>
            <li>Merchant information</li>
            <li>Card usage patterns</li>
            <li>Device and network information</li>
            <li>Behavioral biometrics</li>
        </ul>
        
        <p>Most fraud detection datasets are highly imbalanced, with fraudulent transactions making up less than 1% of all transactions.</p>
    </div>
    """
    
    return explanation

def to_excel(df):
    """
    Convert DataFrame to Excel file for download
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to convert
        
    Returns:
    --------
    bytes
        Excel file as bytes
    """
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        processed_data = output.getvalue()
        return processed_data
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def get_binary_file_downloader_html(bin_data, file_label='File', file_name='data.xlsx'):
    """
    Generate HTML code for file download link
    
    Parameters:
    -----------
    bin_data : bytes
        Binary data
    file_label : str
        Label for download button
    file_name : str
        Name of the file
        
    Returns:
    --------
    str
        HTML code
    """
    if bin_data is None:
        return ""
    
    b64 = base64.b64encode(bin_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{file_label}</a>'
    return href

def display_metrics_table(metrics_dict, model_names):
    """
    Display metrics in a formatted table
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of model metrics
    model_names : list
        List of model names
    """
    if not metrics_dict:
        st.warning("No metrics available to display")
        return
    
    # Create metrics DataFrame
    metrics_data = []
    for model_name in model_names:
        if model_name in metrics_dict:
            metrics = metrics_dict[model_name]
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'AUC': f"{metrics['auc']:.4f}"
            })
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)

def format_number(num):
    """
    Format numbers for display
    
    Parameters:
    -----------
    num : float or int
        Number to format
        
    Returns:
    --------
    str
        Formatted number string
    """
    if isinstance(num, (int, float)):
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}K"
        else:
            return f"{num:.2f}"
    return str(num)
