import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Import our modules
from preprocessing import load_and_inspect_data, preprocess_data, get_feature_names, get_sample_fraud_and_normal
from models import (
    train_logistic_regression, train_decision_tree, train_random_forest, 
    train_xgboost, train_mlp, train_deep_learning, evaluate_model, get_feature_importance
)
from visualization import (
    plot_data_distribution, plot_feature_distributions, plot_confusion_matrix,
    plot_roc_curves, plot_metrics_comparison, plot_all_metrics_comparison,
    plot_feature_importance, plot_training_times, plot_credit_card_security,
    plot_financial_data
)
from utils import (
    get_model_icon, generate_fraud_detection_explanation, 
    generate_sample_data, explain_features, to_excel, 
    get_binary_file_downloader_html, show_classification_report
)
from style import apply_glassy_style, render_header, render_glass_card, render_model_card, render_feature_importance

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_glassy_style()

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = {}
if 'training_times' not in st.session_state:
    st.session_state.training_times = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Render header
render_header()

# Main layout
tabs = st.tabs([
    "📊 Dashboard", 
    "🔍 Data Exploration", 
    "⚙️ Model Training", 
    "📈 Model Comparison", 
    "🛡️ Fraud Detection"
])

# Dashboard tab
with tabs[0]:
    st.write("## Credit Card Fraud Detection Dashboard")
    
    # Dashboard layout - two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; backdrop-filter: blur(5px);">
            <h3 style="margin-top: 0;">Welcome to the Credit Card Fraud Detection System</h3>
            <p>This application uses advanced machine learning algorithms to detect fraudulent credit card transactions.</p>
            <p>Our system employs multiple models for comparison:</p>
            <ul>
                <li>Logistic Regression</li>
                <li>Decision Tree</li>
                <li>Random Forest</li>
                <li>XGBoost</li>
                <li>Deep Learning</li>
            </ul>
            <p>To get started, navigate to the Data Exploration tab to upload your transaction data or use our sample dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display system status
        st.markdown("### System Status")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            data_status = "Loaded" if st.session_state.data is not None else "Not Loaded"
            data_color = "green" if st.session_state.data is not None else "red"
            st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="margin-top: 0;">Data</h4>
                <p style="color: {data_color}; font-size: 1.2em; font-weight: bold;">{data_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col2:
            models_count = len(st.session_state.trained_models)
            models_color = "green" if models_count > 0 else "orange"
            st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="margin-top: 0;">Models Trained</h4>
                <p style="color: {models_color}; font-size: 1.2em; font-weight: bold;">{models_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col3:
            best_model = "None"
            best_accuracy = 0
            
            if st.session_state.model_metrics:
                for model_name, metrics in st.session_state.model_metrics.items():
                    if metrics['accuracy'] > best_accuracy:
                        best_accuracy = metrics['accuracy']
                        best_model = model_name
            
            best_color = "green" if best_accuracy > 0 else "orange"
            accuracy_display = f"{best_accuracy:.2%}" if best_accuracy > 0 else "N/A"
            
            st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="margin-top: 0;">Best Model</h4>
                <p style="color: {best_color}; font-size: 1.2em; font-weight: bold;">{best_model}</p>
                <p>Accuracy: {accuracy_display}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Display fraud detection image
        st.image('https://pixabay.com/get/g14d81335f7ad0c81e96b43cc091924fb6a9c48222b5389d8e67fd67d2f2fa1f9be3a2d6e0451d88dcf064f615853666347b5f673badf346e9bc77b6863191db6_1280.jpg', 
                 caption='Credit Card Security', use_column_width=True)
        
        # Quick access buttons
        st.markdown("### Quick Access")
        
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("📊 Explore Data"):
                st.session_state.active_tab = 1
                st.rerun()
                
            if st.button("🔍 Train Models"):
                st.session_state.active_tab = 2
                st.rerun()
        
        with quick_col2:
            if st.button("📈 Compare Models"):
                st.session_state.active_tab = 3
                st.rerun()
                
            if st.button("🛡️ Detect Fraud"):
                st.session_state.active_tab = 4
                st.rerun()
    
    # Display model cards if models exist
    if st.session_state.trained_models:
        st.markdown("### Trained Models Overview")
        
        # Create columns for model cards
        model_cols = st.columns(len(st.session_state.trained_models))
        
        for i, (model_name, model) in enumerate(st.session_state.trained_models.items()):
            metrics = st.session_state.model_metrics.get(model_name, {})
            accuracy = metrics.get('accuracy', 0)
            
            with model_cols[i]:
                icon = get_model_icon(model_name)
                render_model_card(model_name, "Trained and ready", accuracy, icon)
    
    # Display fraud statistics if data exists
    if st.session_state.data is not None and 'Class' in st.session_state.data.columns:
        st.markdown("### Fraud Statistics")
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Display distribution pie chart
            plot_data_distribution(st.session_state.data, key_suffix="dashboard")
        
        with chart_col2:
            # Display a relevant image or another chart
            plot_financial_data()

# Data Exploration tab
with tabs[1]:
    st.write("## Data Exploration")
    
    upload_col, sample_col = st.columns(2)
    
    with upload_col:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(5px);">
            <h4 style="margin-top: 0;">Upload Transaction Data</h4>
            <p>Upload your credit card transaction dataset (CSV format) for fraud detection.</p>
            <p>The dataset should include transaction features and a 'Class' column (0 for legitimate, 1 for fraud).</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading data..."):
                df, stats = load_and_inspect_data(uploaded_file)
                
                if df is not None:
                    st.session_state.data = df
                    st.success(f"Data loaded successfully: {stats['rows']} rows, {stats['columns']} columns")
    
    with sample_col:
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(5px);">
            <h4 style="margin-top: 0;">Use Sample Data</h4>
            <p>Don't have a dataset? Use our sample credit card transaction data to test the system.</p>
            <p>This will generate a synthetic dataset with both legitimate and fraudulent transactions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data options
        n_samples = st.slider("Number of transactions", 1000, 10000, 3000, 500)
        fraud_ratio = st.slider("Fraud ratio (%)", 0.1, 10.0, 1.0, 0.1) / 100
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                sample_df = generate_sample_data(n_samples, fraud_ratio)
                st.session_state.data = sample_df
                
                stats = {
                    "rows": len(sample_df),
                    "columns": len(sample_df.columns),
                    "fraud_transactions": sample_df['Class'].sum(),
                    "non_fraud_transactions": len(sample_df) - sample_df['Class'].sum(),
                    "fraud_percentage": (sample_df['Class'].sum() / len(sample_df)) * 100
                }
                
                st.success(f"Sample data generated: {stats['rows']} transactions, {stats['fraud_transactions']} fraudulent ({stats['fraud_percentage']:.2f}%)")
    
    # Display data information if data is loaded
    if st.session_state.data is not None:
        # Data overview section
        st.markdown("### Data Overview")
        
        # Display data stats
        df = st.session_state.data
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with stats_col2:
            if 'Class' in df.columns:
                st.metric("Fraudulent Transactions", f"{int(df['Class'].sum()):,}")
        
        with stats_col3:
            if 'Class' in df.columns:
                st.metric("Legitimate Transactions", f"{int(len(df) - df['Class'].sum()):,}")
        
        with stats_col4:
            if 'Class' in df.columns:
                fraud_percentage = (df['Class'].sum() / len(df)) * 100
                st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
        
        # Display data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10))
        
        # Data distribution and features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Transaction Distribution")
            plot_data_distribution(df, key_suffix="data_exploration")
        
        with col2:
            st.markdown("### Features Information")
            st.markdown(explain_features(), unsafe_allow_html=True)
        
        # Feature visualization
        st.markdown("### Feature Distributions")
        plot_feature_distributions(df, key_suffix="data_exploration")
        
        # Sample transactions
        if 'Class' in df.columns:
            st.markdown("### Sample Transactions")
            
            fraud_df, normal_df = get_sample_fraud_and_normal(df)
            
            sample_col1, sample_col2 = st.columns(2)
            
            with sample_col1:
                st.markdown("#### Fraudulent Transactions")
                st.dataframe(fraud_df)
            
            with sample_col2:
                st.markdown("#### Legitimate Transactions")
                st.dataframe(normal_df)
        
        # Download data
        if st.button("Export Data"):
            excel_data = to_excel(df)
            st.markdown(get_binary_file_downloader_html(excel_data, "Excel Data", "credit_card_data.xlsx"), unsafe_allow_html=True)

# Model Training tab
with tabs[2]:
    st.write("## Model Training")
    
    if st.session_state.data is None:
        st.warning("Please load data in the Data Exploration tab first.")
    else:
        # Check if 'Class' column exists
        if 'Class' not in st.session_state.data.columns:
            st.error("The dataset must contain a 'Class' column for fraud detection.")
        else:
            # Preprocessing options
            st.markdown("### Data Preprocessing")
            
            preprocess_col1, preprocess_col2 = st.columns(2)
            
            with preprocess_col1:
                test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
                apply_smote = st.checkbox("Apply SMOTE for class imbalance", value=True)
            
            with preprocess_col2:
                random_state = st.number_input("Random seed", 1, 100, 42)
                scaler_type = st.selectbox("Feature scaling method", ["standard", "robust"])
            
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Preprocess data
                    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
                        st.session_state.data,
                        test_size=test_size,
                        random_state=random_state,
                        apply_smote=apply_smote,
                        scaler_type=scaler_type
                    )
                    
                    if X_train is not None:
                        # Store preprocessed data
                        st.session_state.processed_data = {
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'scaler': scaler,
                            'feature_names': feature_names
                        }
                        
                        st.success("Data preprocessing completed!")
                        
                        # Display data shapes
                        st.markdown("#### Preprocessed Data Shapes")
                        shape_col1, shape_col2 = st.columns(2)
                        
                        with shape_col1:
                            st.markdown(f"**Training set:** {X_train.shape[0]} samples")
                            st.markdown(f"**Training set classes:** {np.bincount(y_train)}")
                        
                        with shape_col2:
                            st.markdown(f"**Test set:** {X_test.shape[0]} samples")
                            st.markdown(f"**Test set classes:** {np.bincount(y_test)}")
            
            # Model training section
            st.markdown("### Model Training")
            
            # Display model training options if data is preprocessed
            if st.session_state.processed_data is not None:
                # Select models to train
                st.markdown("#### Select Models to Train")
                
                model_col1, model_col2, model_col3 = st.columns(3)
                
                with model_col1:
                    train_lr = st.checkbox("Logistic Regression", value=True)
                    train_dt = st.checkbox("Decision Tree", value=True)
                
                with model_col2:
                    train_rf = st.checkbox("Random Forest", value=True)
                    train_xgb = st.checkbox("XGBoost", value=True)
                
                with model_col3:
                    train_dl = st.checkbox("Deep Learning", value=True)
                
                # Model hyperparameters
                with st.expander("Model Hyperparameters"):
                    # Logistic Regression
                    st.markdown("#### Logistic Regression")
                    lr_col1, lr_col2 = st.columns(2)
                    with lr_col1:
                        lr_c = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
                    with lr_col2:
                        lr_max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
                    
                    # Decision Tree
                    st.markdown("#### Decision Tree")
                    dt_col1, dt_col2 = st.columns(2)
                    with dt_col1:
                        dt_max_depth = st.slider("Max Depth", 2, 20, 10, 1)
                    with dt_col2:
                        dt_min_samples = st.slider("Min Samples Split", 2, 20, 2, 1)
                    
                    # Random Forest
                    st.markdown("#### Random Forest")
                    rf_col1, rf_col2 = st.columns(2)
                    with rf_col1:
                        rf_n_estimators = st.slider("Number of Trees", 50, 300, 100, 10)
                    with rf_col2:
                        rf_max_depth = st.slider("RF Max Depth", 2, 20, 10, 1)
                    
                    # XGBoost
                    st.markdown("#### XGBoost")
                    xgb_col1, xgb_col2, xgb_col3 = st.columns(3)
                    with xgb_col1:
                        xgb_n_estimators = st.slider("XGB Number of Trees", 50, 300, 100, 10)
                    with xgb_col2:
                        xgb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                    with xgb_col3:
                        xgb_max_depth = st.slider("XGB Max Depth", 2, 10, 5, 1)
                    
                    # Deep Learning
                    st.markdown("#### Deep Learning")
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        dl_epochs = st.slider("Epochs", 10, 100, 50, 5)
                    with dl_col2:
                        dl_batch_size = st.slider("Batch Size", 16, 128, 32, 8)
                
                # Train models button
                if st.button("Train Selected Models"):
                    # Get preprocessed data
                    X_train = st.session_state.processed_data['X_train']
                    X_test = st.session_state.processed_data['X_test']
                    y_train = st.session_state.processed_data['y_train']
                    y_test = st.session_state.processed_data['y_test']
                    feature_names = st.session_state.processed_data['feature_names']
                    
                    # Initialize or clear model storage
                    st.session_state.trained_models = {}
                    st.session_state.model_metrics = {}
                    st.session_state.feature_importances = {}
                    st.session_state.training_times = {}
                    
                    # Train Logistic Regression
                    if train_lr:
                        with st.spinner("Training Logistic Regression..."):
                            model, train_time = train_logistic_regression(
                                X_train, y_train, 
                                max_iter=lr_max_iter, 
                                C=lr_c
                            )
                            
                            metrics = evaluate_model(model, X_test, y_test)
                            feature_importance = get_feature_importance(model, feature_names, 'logistic_regression')
                            
                            st.session_state.trained_models['Logistic Regression'] = model
                            st.session_state.model_metrics['Logistic Regression'] = metrics
                            st.session_state.feature_importances['Logistic Regression'] = feature_importance
                            st.session_state.training_times['Logistic Regression'] = train_time
                            
                            st.success(f"Logistic Regression trained (Accuracy: {metrics['accuracy']:.4f})")
                    
                    # Train Decision Tree
                    if train_dt:
                        with st.spinner("Training Decision Tree..."):
                            model, train_time = train_decision_tree(
                                X_train, y_train, 
                                max_depth=dt_max_depth, 
                                min_samples_split=dt_min_samples
                            )
                            
                            metrics = evaluate_model(model, X_test, y_test)
                            feature_importance = get_feature_importance(model, feature_names, 'decision_tree')
                            
                            st.session_state.trained_models['Decision Tree'] = model
                            st.session_state.model_metrics['Decision Tree'] = metrics
                            st.session_state.feature_importances['Decision Tree'] = feature_importance
                            st.session_state.training_times['Decision Tree'] = train_time
                            
                            st.success(f"Decision Tree trained (Accuracy: {metrics['accuracy']:.4f})")
                    
                    # Train Random Forest
                    if train_rf:
                        with st.spinner("Training Random Forest..."):
                            model, train_time = train_random_forest(
                                X_train, y_train, 
                                n_estimators=rf_n_estimators, 
                                max_depth=rf_max_depth
                            )
                            
                            metrics = evaluate_model(model, X_test, y_test)
                            feature_importance = get_feature_importance(model, feature_names, 'random_forest')
                            
                            st.session_state.trained_models['Random Forest'] = model
                            st.session_state.model_metrics['Random Forest'] = metrics
                            st.session_state.feature_importances['Random Forest'] = feature_importance
                            st.session_state.training_times['Random Forest'] = train_time
                            
                            st.success(f"Random Forest trained (Accuracy: {metrics['accuracy']:.4f})")
                    
                    # Train XGBoost
                    if train_xgb:
                        with st.spinner("Training XGBoost..."):
                            model, train_time = train_xgboost(
                                X_train, y_train, 
                                n_estimators=xgb_n_estimators, 
                                learning_rate=xgb_learning_rate, 
                                max_depth=xgb_max_depth
                            )
                            
                            metrics = evaluate_model(model, X_test, y_test)
                            feature_importance = get_feature_importance(model, feature_names, 'xgboost')
                            
                            st.session_state.trained_models['XGBoost'] = model
                            st.session_state.model_metrics['XGBoost'] = metrics
                            st.session_state.feature_importances['XGBoost'] = feature_importance
                            st.session_state.training_times['XGBoost'] = train_time
                            
                            st.success(f"XGBoost trained (Accuracy: {metrics['accuracy']:.4f})")
                    
                    # Train Deep Learning
                    if train_dl:
                        with st.spinner("Training Deep Learning model..."):
                            model, train_time = train_deep_learning(
                                X_train, y_train, 
                                hidden_layer_sizes=(128, 64, 32),
                                max_iter=dl_epochs
                            )
                            
                            metrics = evaluate_model(model, X_test, y_test, model_type='sklearn')
                            
                            st.session_state.trained_models['Deep Learning'] = model
                            st.session_state.model_metrics['Deep Learning'] = metrics
                            st.session_state.training_times['Deep Learning'] = train_time
                            
                            st.success(f"Deep Learning model trained (Accuracy: {metrics['accuracy']:.4f})")
                    
                    st.success("All selected models have been trained successfully!")
            else:
                st.info("Please preprocess the data first.")

# Model Comparison tab
with tabs[3]:
    st.write("## Model Comparison")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models in the Model Training tab first.")
    else:
        # Get list of trained models
        model_names = list(st.session_state.trained_models.keys())
        metrics_list = [st.session_state.model_metrics[name] for name in model_names]
        
        # Display training times
        st.markdown("### Training Performance")
        times = [st.session_state.training_times[name] for name in model_names]
        plot_training_times(times, model_names)
        
        # Overall performance visualization
        st.markdown("### Overall Model Performance")
        plot_all_metrics_comparison(metrics_list, model_names)
        
        # Display individual metrics
        st.markdown("### Performance Metrics Comparison")
        
        metric_options = ["accuracy", "precision", "recall", "f1_score", "auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            selected_metric = st.selectbox("Select Metric", metric_options, 
                                          format_func=lambda x: metric_labels[metric_options.index(x)])
        
        plot_metrics_comparison(metrics_list, model_names, selected_metric)
        
        # ROC curves
        st.markdown("### ROC Curves")
        plot_roc_curves(metrics_list, model_names)
        
        # Confusion matrices
        st.markdown("### Confusion Matrices")
        
        # Select model for confusion matrix
        cm_model = st.selectbox("Select Model for Confusion Matrix", model_names)
        
        # Display confusion matrix
        if cm_model in st.session_state.model_metrics:
            cm = st.session_state.model_metrics[cm_model]['confusion_matrix']
            plot_confusion_matrix(cm, cm_model)
        
        # Feature importance
        st.markdown("### Feature Importance Analysis")
        
        # Select model for feature importance
        fi_models = [name for name in model_names 
                    if name in st.session_state.feature_importances and st.session_state.feature_importances[name] is not None]
        
        if fi_models:
            fi_model = st.selectbox("Select Model for Feature Importance", fi_models)
            
            # Display feature importance
            if fi_model in st.session_state.feature_importances:
                importance_dict = st.session_state.feature_importances[fi_model]
                if importance_dict:
                    plot_feature_importance(importance_dict, fi_model)
        else:
            st.info("Feature importance not available for the trained models.")
        
        # Detailed classification reports
        st.markdown("### Detailed Classification Reports")
        
        # Select model for classification report
        cr_model = st.selectbox("Select Model for Classification Report", model_names, key="cr_model")
        
        # Display classification report
        if cr_model in st.session_state.model_metrics:
            report = st.session_state.model_metrics[cr_model]['classification_report']
            show_classification_report(report, cr_model)

# Fraud Detection tab
with tabs[4]:
    st.write("## Fraud Detection Demo")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models in the Model Training tab first.")
    else:
        # Get list of trained models
        model_names = list(st.session_state.trained_models.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### How Models Detect Credit Card Fraud")
            
            # Select model for explanation
            explanation_model = st.selectbox("Select Model", model_names)
            
            # Display model explanation
            explanation = generate_fraud_detection_explanation(explanation_model)
            st.markdown(explanation, unsafe_allow_html=True)
            
            # Display model icon
            icon = get_model_icon(explanation_model)
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 20px 0;">
                <div style="margin-right: 15px;">{icon}</div>
                <h3 style="margin: 0;">{explanation_model}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics for the selected model
            if explanation_model in st.session_state.model_metrics:
                metrics = st.session_state.model_metrics[explanation_model]
                st.markdown("#### Performance Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                
                with metric_col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                
                with metric_col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                
                with metric_col4:
                    st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        
        with col2:
            # Display security image
            st.image('https://pixabay.com/get/g443db24863265c3a7f46eca283775d3ca0f51b4ca64b36201675bd93f406a63ef733567949e9f7eb80fa2350ad40f7d3ceb50fefd41c2011f43287ab81d109b2_1280.jpg', 
                     caption='Credit Card Security', use_column_width=True)
        
        # Interactive detection demo
        st.markdown("### Interactive Fraud Detection Demo")
        
        if st.session_state.processed_data is None:
            st.warning("Please preprocess data in the Model Training tab first.")
        else:
            # Get feature names
            feature_names = st.session_state.processed_data['feature_names']
            scaler = st.session_state.processed_data['scaler']
            
            # Select model for prediction
            pred_model = st.selectbox("Select Model for Prediction", model_names, key="pred_model")
            
            # Two options: random sample or manual input
            demo_option = st.radio("Demo Option", ["Random Transaction Sample", "Manual Feature Input"])
            
            if demo_option == "Random Transaction Sample":
                # Get random sample from test data
                if 'X_test' in st.session_state.processed_data:
                    X_test = st.session_state.processed_data['X_test']
                    y_test = st.session_state.processed_data['y_test']
                    
                    # Select random index
                    random_idx = np.random.randint(0, len(X_test))
                    random_features = X_test[random_idx]
                    true_label = y_test[random_idx]
                    
                    # Display true label
                    st.markdown(f"#### True Label: {'Fraudulent' if true_label == 1 else 'Legitimate'}")
                    
                    # Make prediction
                    model = st.session_state.trained_models[pred_model]
                    
                    if pred_model == "Deep Learning":
                        pred_proba = model.predict(random_features.reshape(1, -1))[0][0]
                        prediction = 1 if pred_proba > 0.5 else 0
                    else:
                        prediction = model.predict(random_features.reshape(1, -1))[0]
                        pred_proba = model.predict_proba(random_features.reshape(1, -1))[0][1]
                    
                    # Display prediction
                    is_correct = prediction == true_label
                    
                    st.markdown(f"""
                    <div style="background-color: rgba({','.join(['39, 174, 96' if is_correct else '231, 76, 60'])}, 0.2); 
                                padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4 style="margin-top: 0;">Prediction Result</h4>
                        <p style="font-size: 1.5em; font-weight: bold;">
                            {'✅' if is_correct else '❌'} {pred_model} predicts this transaction is 
                            <span style="color: {'#27AE60' if prediction == 0 else '#E74C3C'};">
                                {'Legitimate' if prediction == 0 else 'Fraudulent'}
                            </span>
                            with {pred_proba:.2%} confidence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display feature values
                    st.markdown("#### Transaction Features")
                    
                    # Show top important features if available
                    if pred_model in st.session_state.feature_importances and st.session_state.feature_importances[pred_model]:
                        importance_dict = st.session_state.feature_importances[pred_model]
                        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        top_features = [f[0] for f in sorted_features[:10]]
                        
                        feature_values = {}
                        for i, feature in enumerate(feature_names):
                            if feature in top_features:
                                feature_values[feature] = random_features[i]
                        
                        # Display feature values in columns
                        feature_cols = st.columns(5)
                        for i, (feature, value) in enumerate(feature_values.items()):
                            with feature_cols[i % 5]:
                                st.metric(feature, f"{value:.4f}")
            else:
                # Manual feature input
                st.markdown("#### Enter Feature Values")
                
                if len(feature_names) > 10:
                    st.info("For simplicity, only the top 10 features are shown for manual input.")
                
                # Get top features if feature importance is available
                top_features = feature_names[:10]  # Default
                
                if pred_model in st.session_state.feature_importances and st.session_state.feature_importances[pred_model]:
                    importance_dict = st.session_state.feature_importances[pred_model]
                    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    top_features = [f[0] for f in sorted_features[:10]]
                
                # Create input fields for features
                manual_features = {}
                
                # Create columns for input
                input_cols = st.columns(2)
                
                for i, feature in enumerate(top_features):
                    with input_cols[i % 2]:
                        manual_features[feature] = st.number_input(
                            feature, 
                            value=0.0, 
                            format="%.4f"
                        )
                
                # Create full feature vector
                full_features = np.zeros(len(feature_names))
                
                for i, feature in enumerate(feature_names):
                    if feature in manual_features:
                        full_features[i] = manual_features[feature]
                
                # Scale features
                scaled_features = scaler.transform(full_features.reshape(1, -1))[0]
                
                # Make prediction button
                if st.button("Predict"):
                    # Make prediction
                    model = st.session_state.trained_models[pred_model]
                    
                    if pred_model == "Deep Learning":
                        pred_proba = model.predict(scaled_features.reshape(1, -1))[0][0]
                        prediction = 1 if pred_proba > 0.5 else 0
                    else:
                        prediction = model.predict(scaled_features.reshape(1, -1))[0]
                        pred_proba = model.predict_proba(scaled_features.reshape(1, -1))[0][1]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div style="background-color: rgba({','.join(['39, 174, 96' if prediction == 0 else '231, 76, 60'])}, 0.2); 
                                padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4 style="margin-top: 0;">Prediction Result</h4>
                        <p style="font-size: 1.5em; font-weight: bold;">
                            {pred_model} predicts this transaction is 
                            <span style="color: {'#27AE60' if prediction == 0 else '#E74C3C'};">
                                {'Legitimate' if prediction == 0 else 'Fraudulent'}
                            </span>
                            with {pred_proba:.2%} confidence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Features that indicate fraud
        st.markdown("### Key Indicators of Credit Card Fraud")
        
        # Create feature importance visualization if available
        if any(st.session_state.feature_importances.values()):
            # Find model with highest accuracy
            best_model = max(st.session_state.model_metrics.items(), 
                            key=lambda x: x[1]['accuracy'])[0]
            
            # Get feature importance for best model
            if best_model in st.session_state.feature_importances and st.session_state.feature_importances[best_model]:
                importance_dict = st.session_state.feature_importances[best_model]
                render_feature_importance(feature_names, list(importance_dict.values()))
            else:
                # Get first available feature importance
                for model, importance in st.session_state.feature_importances.items():
                    if importance:
                        render_feature_importance(feature_names, list(importance.values()))
                        break
        else:
            st.info("Feature importance not available for the trained models.")
        
        # Prevention tips
        st.markdown("### Credit Card Fraud Prevention Tips")
        
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; backdrop-filter: blur(5px); margin-top: 20px;">
            <h4 style="margin-top: 0;">How to Protect Against Credit Card Fraud</h4>
            
            <ul>
                <li><strong>Monitor your accounts regularly</strong> - Check transaction history frequently for suspicious activity</li>
                <li><strong>Use strong, unique passwords</strong> - Never use the same password across multiple financial accounts</li>
                <li><strong>Enable two-factor authentication</strong> - Add an extra layer of security beyond passwords</li>
                <li><strong>Be cautious with public Wi-Fi</strong> - Avoid accessing financial accounts on unsecured networks</li>
                <li><strong>Keep your personal information private</strong> - Be wary of phishing attempts requesting card details</li>
                <li><strong>Set up transaction alerts</strong> - Get notified of purchases over a certain amount</li>
                <li><strong>Use EMV chip-enabled cards</strong> - They provide better security than magnetic stripe cards</li>
                <li><strong>Check ATMs and card readers for skimming devices</strong> - Look for anything unusual before inserting your card</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
