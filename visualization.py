import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_data_distribution(df, key_suffix="main"):
    """
    Plot the distribution of fraud vs. non-fraud transactions
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    key_suffix : str
        Suffix for the key to ensure uniqueness
    """
    if 'Class' not in df.columns:
        st.warning("No 'Class' column found in data")
        return
    
    # Count fraud vs. non-fraud
    class_counts = df['Class'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    class_counts['Class'] = class_counts['Class'].map({0: 'Legitimate', 1: 'Fraud'})
    
    # Create pie chart
    fig = px.pie(
        class_counts, 
        values='Count', 
        names='Class',
        title='Transaction Distribution',
        color='Class',
        color_discrete_map={'Legitimate': '#2E86C1', 'Fraud': '#E74C3C'},
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"data_distribution_pie_{key_suffix}")

def plot_feature_distributions(df, n_features=6, key_suffix="main"):
    """
    Plot distributions of top features
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    n_features : int
        Number of features to plot
    key_suffix : str
        Suffix for the key to ensure uniqueness
    """
    if 'Class' not in df.columns or df is None:
        return
    
    # Get numeric columns excluding Class and Time
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Class' in numeric_cols:
        numeric_cols.remove('Class')
    if 'Time' in numeric_cols:
        numeric_cols.remove('Time')
    
    if len(numeric_cols) == 0:
        st.warning("No numeric features found for visualization")
        return
    
    # Calculate variance of each feature and select top features
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.index[:min(n_features, len(numeric_cols))].tolist()
    
    # Create subplots
    rows = len(top_features) // 2 + (len(top_features) % 2)
    cols = min(2, len(top_features))
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=top_features)
    
    for i, feature in enumerate(top_features):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Create histogram for legitimate transactions
        fig.add_trace(
            go.Histogram(
                x=df[df['Class'] == 0][feature],
                name='Legitimate',
                marker_color='#2E86C1',
                opacity=0.7,
                nbinsx=30,
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
        
        # Create histogram for fraudulent transactions
        fig.add_trace(
            go.Histogram(
                x=df[df['Class'] == 1][feature],
                name='Fraud',
                marker_color='#E74C3C',
                opacity=0.7,
                nbinsx=30,
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=max(400, rows * 200),
        title_text='Feature Distributions: Fraud vs Legitimate',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"feature_distributions_{key_suffix}")

def plot_confusion_matrix(cm, model_name):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : ndarray
        Confusion matrix
    model_name : str
        Name of the model
    """
    # Calculate derived metrics
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    
    # Create annotation text
    z_text = [
        [f'TN: {tn}', f'FP: {fp}'],
        [f'FN: {fn}', f'TP: {tp}']
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Legitimate', 'Predicted Fraud'],
        y=['Actual Legitimate', 'Actual Fraud'],
        colorscale='Reds',
        showscale=False,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 14}
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_name} (Accuracy: {accuracy:.4f})",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"confusion_matrix_{model_name.replace(' ', '_')}")

def plot_metrics_comparison(models_metrics, model_names, metric='accuracy'):
    """
    Plot comparison of a specific metric across models
    
    Parameters:
    -----------
    models_metrics : list
        List of model metrics dictionaries
    model_names : list
        List of model names
    metric : str
        Metric to compare
    """
    values = [metrics[metric] for metrics in models_metrics]
    colors = ['#E74C3C', '#2E86C1', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=values,
        marker_color=colors[:len(model_names)],
        text=[f"{val:.4f}" for val in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"{metric.capitalize()} Comparison",
        xaxis_title="Model",
        yaxis_title=metric.capitalize(),
        yaxis=dict(range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"metrics_comparison_{metric}")

def plot_all_metrics_comparison(models_metrics, model_names):
    """
    Plot comparison of all metrics across models
    
    Parameters:
    -----------
    models_metrics : list
        List of model metrics dictionaries
    model_names : list
        List of model names
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    # Prepare data
    data = {}
    for metric in metrics:
        data[metric] = [metrics_dict[metric] for metrics_dict in models_metrics]
    
    # Create figure
    fig = go.Figure()
    
    colors = ['#E74C3C', '#2E86C1', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
    
    for i, model in enumerate(model_names):
        values = [data[metric][i] for metric in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_names,
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Model Performance Comparison",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key="all_metrics_comparison")

def plot_feature_importance(importance_dict, model_name, top_n=10):
    """
    Plot feature importance
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary with feature importance
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    """
    if importance_dict is None or len(importance_dict) == 0:
        st.warning(f"Feature importance not available for {model_name}")
        return
    
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N features
    top_features = sorted_features[:top_n]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=importances,
        orientation='h',
        marker_color='#E74C3C'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance - {model_name}",
        xaxis_title="Importance",
        yaxis_title="Feature",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{model_name.replace(' ', '_')}")

def plot_training_times(training_times, model_names):
    """
    Plot model training times
    
    Parameters:
    -----------
    training_times : list
        List of training times
    model_names : list
        List of model names
    """
    fig = go.Figure()
    
    colors = ['#E74C3C', '#2E86C1', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=training_times,
        marker_color=colors[:len(model_names)],
        text=[f"{time:.2f}s" for time in training_times],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Model Training Times",
        xaxis_title="Model",
        yaxis_title="Time (seconds)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True, key="training_times")
