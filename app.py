import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Import custom modules
from preprocessing import load_and_inspect_data, preprocess_data, get_feature_names
from models import (
    train_logistic_regression, train_decision_tree, train_random_forest,
    train_xgboost, train_deep_learning, evaluate_model, get_feature_importance
)
from visualization import (
    plot_data_distribution, plot_feature_distributions, plot_confusion_matrix,
    plot_metrics_comparison, plot_all_metrics_comparison, plot_feature_importance,
    plot_training_times
)
from style import apply_glassy_style, render_header, render_glass_card, render_model_card
from utils import (
    get_model_icon, generate_fraud_detection_explanation, generate_sample_data,
    explain_features, to_excel
)

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_glassy_style()

def main():
    """Main application function"""
    
    # Render header
    render_header()
    
    # Modern Sidebar Navigation
    with st.sidebar:
        # Custom CSS for modern blue sidebar
        st.markdown("""
        <style>
        /* Force sidebar background */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4A90E2 0%, #357ABD 100%) !important;
        }
        
        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #4A90E2 0%, #357ABD 100%) !important;
        }
        
        /* Sidebar Header */
        .sidebar-header {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px 15px;
            margin: -1rem -1rem 2rem -1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Section headers */
        .section-header {
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            margin: 25px 0 15px 0;
            padding-left: 5px;
        }
        
        /* Override Streamlit selectbox styling */
        .stSelectbox > div > div {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border: none !important;
            border-radius: 8px !important;
            color: #357ABD !important;
            font-weight: 500 !important;
        }
        
        .stSelectbox > div > div:hover {
            background-color: rgba(255, 255, 255, 1) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stSelectbox label {
            color: white !important;
            font-weight: 500 !important;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            padding: 15px 10px !important;
            margin-top: 10px !important;
        }
        
        .stRadio > div > label {
            color: white !important;
            font-weight: 500 !important;
        }
        
        .stRadio > div > div > label {
            color: white !important;
        }
        
        .stRadio > div > div > label > div {
            color: white !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px dashed rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px !important;
            padding: 20px !important;
        }
        
        .stFileUploader label {
            color: white !important;
            font-weight: 500 !important;
        }
        
        .stFileUploader > div > div {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        
        /* Slider styling */
        .stSlider > div > div > div {
            color: white !important;
        }
        
        .stSlider label {
            color: white !important;
            font-weight: 500 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #357ABD !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background: rgba(255, 255, 255, 1) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar Header
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="color: white; margin: 0; font-weight: 600;">Sidebar</h2>
            <div style="width: 60px; height: 2px; background: white; margin-top: 8px; opacity: 0.8;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "📈 Results", "🔍 Fraud Detection"]
        )
        
        # Data Source Section
        st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
        
        data_source = st.radio(
            "Select data source:",
            ["Upload CSV File", "Generate Sample Data"]
        )
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'stats' not in st.session_state:
            st.session_state.stats = None
        if 'models' not in st.session_state:
            st.session_state.models = {}
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
        if 'training_times' not in st.session_state:
            st.session_state.training_times = {}
        
        # Handle data loading with clean styling
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload a credit card transaction dataset with a 'Class' column"
            )
            
            if uploaded_file is not None:
                if st.session_state.data is None or st.button("Reload Data"):
                    with st.spinner("Loading data..."):
                        st.session_state.data, st.session_state.stats = load_and_inspect_data(uploaded_file)
                        if st.session_state.data is not None:
                            st.success("Data loaded successfully!")
        
        else:  # Generate Sample Data
            st.markdown('<div class="section-header">Parameters</div>', unsafe_allow_html=True)
            
            n_samples = st.slider("Number of samples", 1000, 10000, 5000, 500)
            fraud_ratio = st.slider("Fraud ratio", 0.01, 0.2, 0.1, 0.01)
            
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    st.session_state.data = generate_sample_data(n_samples, fraud_ratio)
                    st.session_state.stats = {
                        "rows": len(st.session_state.data),
                        "columns": len(st.session_state.data.columns),
                        "missing_values": st.session_state.data.isnull().sum().sum(),
                        "duplicate_rows": st.session_state.data.duplicated().sum(),
                        "fraud_transactions": st.session_state.data['Class'].sum(),
                        "non_fraud_transactions": len(st.session_state.data) - st.session_state.data['Class'].sum(),
                        "fraud_percentage": (st.session_state.data['Class'].sum() / len(st.session_state.data)) * 100
                    }
                    st.success("Sample data generated successfully!")
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "📈 Results":
        show_results_page()
    elif page == "🔍 Fraud Detection":
        show_fraud_detection_page()

def show_home_page():
    """Display the home page"""
    st.title("Welcome to Credit Card Fraud Detection System")
    
    # Main description
    st.markdown("""
    ### 🛡️ Advanced Machine Learning Protection
    
    This comprehensive fraud detection system employs multiple machine learning algorithms 
    to identify potentially fraudulent credit card transactions with high accuracy.
    """)
    
    # Navigation cards for all pages
    st.markdown("### 📋 Available Pages")
    
    # Create 2x2 grid layout for page cards
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0; backdrop-filter: blur(5px); border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-top: 0; color: #2E86C1;">📊 Data Analysis</h4>
            <p style="color: rgba(255,255,255,0.8); margin-bottom: 15px;">Upload your CSV data or generate sample datasets. Explore data distributions and view transaction samples.</p>
            <ul style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                <li>Upload CSV files</li>
                <li>Generate sample data</li>
                <li>View data statistics</li>
                <li>Explore distributions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with row1_col2:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0; backdrop-filter: blur(5px); border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-top: 0; color: #27AE60;">🤖 Model Training</h4>
            <p style="color: rgba(255,255,255,0.8); margin-bottom: 15px;">Train multiple machine learning models and compare their performance on your fraud detection dataset.</p>
            <ul style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                <li>5 ML algorithms</li>
                <li>Hyperparameter tuning</li>
                <li>SMOTE balancing</li>
                <li>Training progress</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with row2_col1:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0; backdrop-filter: blur(5px); border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-top: 0; color: #F39C12;">📈 Results</h4>
            <p style="color: rgba(255,255,255,0.8); margin-bottom: 15px;">View comprehensive model performance metrics, confusion matrices, and feature importance analysis.</p>
            <ul style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                <li>Performance metrics</li>
                <li>Confusion matrices</li>
                <li>Feature importance</li>
                <li>Model comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with row2_col2:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0; backdrop-filter: blur(5px); border: 1px solid rgba(255, 255, 255, 0.1);">
            <h4 style="margin-top: 0; color: #E74C3C;">🔍 Fraud Detection</h4>
            <p style="color: rgba(255,255,255,0.8); margin-bottom: 15px;">Use trained models to detect fraud in real-time. Input transaction data and get instant fraud probability scores.</p>
            <ul style="color: rgba(255,255,255,0.7); font-size: 0.9em;">
                <li>Real-time analysis</li>
                <li>Fraud probability scoring</li>
                <li>Model explanations</li>
                <li>Transaction classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # System status section
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if st.session_state.data is not None:
            st.success("✅ Data Loaded")
            st.info(f"📁 {st.session_state.stats['rows']:,} transactions")
            st.info(f"🚨 {st.session_state.stats['fraud_percentage']:.2f}% fraud rate")
        else:
            st.warning("⚠️ No data loaded")
    
    with status_col2:
        if st.session_state.models:
            st.success(f"🤖 {len(st.session_state.models)} models trained")
            best_model = max(st.session_state.metrics.items(), key=lambda x: x[1]['accuracy'])
            st.info(f"🏆 Best: {best_model[0]} ({best_model[1]['accuracy']:.2%})")
        else:
            st.info("🔄 No models trained yet")
    
    with status_col3:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h4 style="margin-top: 0;">Quick Start</h4>
            <p style="margin: 0; color: rgba(255,255,255,0.8);">Use the sidebar to select a page and get started with your fraud detection analysis!</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_analysis_page():
    """Display the data analysis page"""
    st.title("📊 Data Analysis & Exploration")
    
    if st.session_state.data is None:
        st.warning("Please load data first using the sidebar.")
        return
    
    df = st.session_state.data
    stats = st.session_state.stats
    
    # Display data statistics
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{stats['rows']:,}")
    with col2:
        st.metric("Features", stats['columns'])
    with col3:
        st.metric("Fraud Cases", f"{stats['fraud_transactions']:,}")
    with col4:
        st.metric("Fraud Rate", f"{stats['fraud_percentage']:.2f}%")
    
    # Data quality metrics
    st.subheader("Data Quality")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Missing Values", stats['missing_values'])
    with col2:
        st.metric("Duplicate Rows", stats['duplicate_rows'])
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Feature Analysis", "Sample Data"])
    
    with tab1:
        plot_data_distribution(df, "analysis")
    
    with tab2:
        plot_feature_distributions(df, 6, "analysis")
    
    with tab3:
        st.subheader("Sample Transactions")
        
        # Show sample legitimate and fraud transactions
        if 'Class' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Legitimate Transactions (Sample)**")
                legitimate_sample = df[df['Class'] == 0].head(5)
                st.dataframe(legitimate_sample, use_container_width=True)
            
            with col2:
                st.markdown("**Fraudulent Transactions (Sample)**")
                fraud_sample = df[df['Class'] == 1].head(5)
                if len(fraud_sample) > 0:
                    st.dataframe(fraud_sample, use_container_width=True)
                else:
                    st.info("No fraudulent transactions in the first few rows")

def show_model_training_page():
    """Display the model training page"""
    st.title("🤖 Machine Learning Model Training")
    
    if st.session_state.data is None:
        st.warning("Please load data first using the sidebar.")
        return
    
    df = st.session_state.data
    
    # Model selection
    st.subheader("Model Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_models = st.multiselect(
            "Choose models to train:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Deep Learning"],
            default=["Logistic Regression", "Random Forest"]
        )
    
    with col2:
        st.subheader("Training Options")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        apply_smote = st.checkbox("Apply SMOTE (balancing)", value=True)
        scaler_type = st.selectbox("Scaler type", ["standard", "robust"])
    
    # Model training
    if st.button("🚀 Train Selected Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return
        
        with st.spinner("Preprocessing data..."):
            # Preprocess data
            X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
                df, test_size=test_size, apply_smote=apply_smote, scaler_type=scaler_type
            )
            
            if X_train is None:
                st.error("Error in data preprocessing.")
                return
        
        # Store preprocessed data in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_names
        
        # Train models
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        models = {}
        metrics = {}
        training_times = {}
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            try:
                # Train model
                if model_name == "Logistic Regression":
                    model, train_time = train_logistic_regression(X_train, y_train)
                elif model_name == "Decision Tree":
                    model, train_time = train_decision_tree(X_train, y_train)
                elif model_name == "Random Forest":
                    model, train_time = train_random_forest(X_train, y_train)
                elif model_name == "XGBoost":
                    model, train_time = train_xgboost(X_train, y_train)
                elif model_name == "Deep Learning":
                    model, train_time = train_deep_learning(X_train, y_train)
                
                # Evaluate model
                model_metrics = evaluate_model(model, X_test, y_test)
                
                # Store results
                models[model_name] = model
                metrics[model_name] = model_metrics
                training_times[model_name] = train_time
                
                st.success(f"✅ {model_name} trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        # Update session state
        st.session_state.models.update(models)
        st.session_state.metrics.update(metrics)
        st.session_state.training_times.update(training_times)
        
        status_text.text("Training completed!")
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"🎉 Successfully trained {len(models)} models!")

def show_results_page():
    """Display the results and model comparison page"""
    st.title("📈 Model Performance Results")
    
    if not st.session_state.models:
        st.warning("No trained models found. Please train models first.")
        return
    
    models = st.session_state.models
    metrics = st.session_state.metrics
    training_times = st.session_state.training_times
    
    # Model performance summary
    st.subheader("Model Performance Summary")
    
    # Create summary table
    summary_data = []
    for model_name in models.keys():
        if model_name in metrics:
            model_metrics = metrics[model_name]
            summary_data.append({
                "Model": model_name,
                "Accuracy": f"{model_metrics['accuracy']:.4f}",
                "Precision": f"{model_metrics['precision']:.4f}",
                "Recall": f"{model_metrics['recall']:.4f}",
                "F1 Score": f"{model_metrics['f1_score']:.4f}",
                "AUC": f"{model_metrics['auc']:.4f}",
                "Training Time": f"{training_times.get(model_name, 0):.2f}s"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Visualizations
    st.subheader("Performance Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Metrics Comparison", "Confusion Matrices", "Feature Importance", "Training Times"])
    
    with tab1:
        # Metrics comparison
        if len(metrics) > 1:
            plot_all_metrics_comparison(list(metrics.values()), list(metrics.keys()))
        
        # Individual metric comparisons
        metric_cols = st.columns(2)
        with metric_cols[0]:
            plot_metrics_comparison(list(metrics.values()), list(metrics.keys()), 'accuracy')
        with metric_cols[1]:
            plot_metrics_comparison(list(metrics.values()), list(metrics.keys()), 'f1_score')
    
    with tab2:
        # Confusion matrices
        for model_name, model_metrics in metrics.items():
            plot_confusion_matrix(model_metrics['confusion_matrix'], model_name)
    
    with tab3:
        # Feature importance
        if hasattr(st.session_state, 'feature_names'):
            feature_names = st.session_state.feature_names
            for model_name, model in models.items():
                model_type = model_name.lower().replace(' ', '_')
                importance = get_feature_importance(model, feature_names, model_type)
                if importance:
                    plot_feature_importance(importance, model_name)
    
    with tab4:
        # Training times
        if training_times:
            plot_training_times(list(training_times.values()), list(training_times.keys()))

def show_fraud_detection_page():
    """Display the fraud detection interface"""
    st.title("🔍 Real-time Fraud Detection")
    
    if not st.session_state.models:
        st.warning("No trained models found. Please train models first.")
        return
    
    st.subheader("Transaction Analysis")
    
    # Model selection for prediction
    selected_model = st.selectbox(
        "Choose model for prediction:",
        list(st.session_state.models.keys())
    )
    
    if selected_model not in st.session_state.models:
        st.error("Selected model not available.")
        return
    
    # Input methods
    input_method = st.radio(
        "Input method:",
        ["Manual Input", "Random Sample from Dataset"]
    )
    
    if input_method == "Manual Input":
        st.info("For this demo, manual input requires knowledge of the exact feature structure. Using random sample is recommended.")
        
        # This would require a complex form for all features
        # For now, we'll show this as a placeholder
        st.text_area(
            "Feature values (comma-separated):",
            placeholder="Enter feature values separated by commas...",
            help="This requires exact knowledge of feature structure and scaling"
        )
        
        if st.button("Analyze Transaction"):
            st.warning("Manual input analysis requires implementation of feature parsing and scaling.")
    
    else:  # Random Sample
        if st.session_state.data is not None and hasattr(st.session_state, 'X_test'):
            if st.button("🎲 Analyze Random Transaction", type="primary"):
                # Get random sample from test set
                random_idx = np.random.randint(0, len(st.session_state.X_test))
                sample_features = st.session_state.X_test[random_idx:random_idx+1]
                actual_class = st.session_state.y_test.iloc[random_idx] if hasattr(st.session_state.y_test, 'iloc') else st.session_state.y_test[random_idx]
                
                # Make prediction
                model = st.session_state.models[selected_model]
                prediction = model.predict(sample_features)[0]
                prediction_proba = model.predict_proba(sample_features)[0]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Actual Class",
                        "Fraud" if actual_class == 1 else "Legitimate",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Predicted Class",
                        "Fraud" if prediction == 1 else "Legitimate",
                        delta="Correct" if prediction == actual_class else "Incorrect"
                    )
                
                with col3:
                    fraud_probability = prediction_proba[1] * 100
                    st.metric(
                        "Fraud Probability",
                        f"{fraud_probability:.2f}%",
                        delta=None
                    )
                
                # Risk assessment
                if fraud_probability > 80:
                    st.error("🚨 HIGH RISK - Immediate attention required!")
                elif fraud_probability > 50:
                    st.warning("⚠️ MEDIUM RISK - Review recommended")
                else:
                    st.success("✅ LOW RISK - Transaction appears legitimate")
                
                # Model explanation
                st.subheader("Model Explanation")
                explanation = generate_fraud_detection_explanation(selected_model)
                st.markdown(explanation, unsafe_allow_html=True)
        
        else:
            st.warning("Please ensure data is loaded and models are trained with test data available.")

if __name__ == "__main__":
    main()
