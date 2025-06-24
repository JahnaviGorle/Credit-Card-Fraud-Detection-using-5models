# Credit Card Fraud Detection System

## Overview

This is a comprehensive machine learning application for credit card fraud detection built with Streamlit. The system provides a complete end-to-end solution for analyzing credit card transaction data, training multiple ML models, and detecting fraudulent transactions through an intuitive web interface.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **UI Design**: Custom glassy/glass morphism styling with dark theme
- **Navigation**: Multi-page application with sidebar navigation
- **Visualization**: Plotly for interactive charts and Matplotlib for static plots
- **Responsive Layout**: Wide layout with expandable sidebar

### Backend Architecture
- **Language**: Python 3.11
- **Structure**: Modular design with separate files for different concerns:
  - `app.py`: Main application entry point and UI orchestration
  - `models.py`: Machine learning model implementations
  - `preprocessing.py`: Data preprocessing and feature engineering
  - `visualization.py`: Chart and graph generation
  - `utils.py`: Utility functions and helpers
  - `style.py`: CSS styling and UI components

### Machine Learning Pipeline
- **Models Supported**: 
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Deep Learning (MLP/Neural Networks)
- **Data Processing**: StandardScaler/RobustScaler for feature scaling
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, ROC-AUC

## Key Components

### Data Processing Module (`preprocessing.py`)
- **Purpose**: Handle data loading, inspection, and preprocessing
- **Features**: 
  - CSV file upload and validation
  - Missing value detection
  - Feature scaling options
  - Train-test split functionality
  - SMOTE oversampling for imbalanced datasets

### Model Training Module (`models.py`)
- **Purpose**: Implement and train various ML algorithms
- **Features**:
  - Multiple algorithm implementations
  - Hyperparameter customization
  - Training time tracking
  - Model evaluation metrics calculation

### Visualization Module (`visualization.py`)
- **Purpose**: Generate interactive and static visualizations
- **Features**:
  - Data distribution plots
  - Feature importance visualizations
  - Confusion matrices
  - ROC curves
  - Model performance comparisons

### Utility Module (`utils.py`)
- **Purpose**: Provide helper functions and utilities
- **Features**:
  - Model icons and branding
  - Data export functionality
  - Sample data generation
  - Feature explanation utilities

### Styling Module (`style.py`)
- **Purpose**: Custom UI styling and theming
- **Features**:
  - Glass morphism design system
  - Custom CSS for Streamlit components
  - Responsive card layouts
  - Dark theme with blue/red accent colors

## Data Flow

1. **Data Input**: Users upload CSV files containing credit card transaction data
2. **Data Inspection**: System analyzes data structure, missing values, and class distribution
3. **Preprocessing**: Data is cleaned, scaled, and split into training/testing sets
4. **Model Training**: Multiple ML models are trained with user-specified parameters
5. **Evaluation**: Models are evaluated using various metrics and visualizations
6. **Prediction**: Trained models can be used for fraud detection on new transactions
7. **Export**: Results and trained models can be exported for further use

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Primary machine learning framework for traditional algorithms
- **XGBoost**: Gradient boosting implementation
- **imbalanced-learn**: SMOTE implementation for handling class imbalance

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Visualization
- **plotly**: Interactive plotting library
- **matplotlib**: Static plotting (backup/additional charts)

### Web Framework
- **streamlit**: Web application framework

### Utilities
- **xlsxwriter**: Excel file export functionality

## Deployment Strategy

### Platform
- **Replit**: Cloud-based deployment platform
- **Nix Environment**: Reproducible development environment with Python 3.11
- **System Packages**: Cairo, FFmpeg, FreeType, GhostScript for advanced graphics support

### Configuration
- **Port**: Application runs on port 5000
- **Deployment Target**: Autoscale for handling variable traffic
- **Server Configuration**: Headless mode with custom theme settings

### Environment Setup
- **Python Version**: 3.11+ required
- **Package Management**: UV lock file for dependency resolution
- **System Dependencies**: Graphics libraries for chart rendering

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- June 24, 2025: Initial setup complete
- June 24, 2025: Removed Credit Card Fraud Detection Features section from home page per user request
- June 24, 2025: Fixed Python 3.11 compatibility issues and installed all required ML dependencies
- June 24, 2025: Enhanced home page to display all available pages with detailed descriptions and features
- June 24, 2025: Enhanced navigation bar with beautiful gradient styling, hover effects, and improved visual design
- June 24, 2025: Replaced dropdown navigation with individual clickable page items with hover selection and active state indicators
- June 24, 2025: Removed decorative line under sidebar header for cleaner appearance
- June 24, 2025: Removed sidebar header section completely for minimal design
- June 24, 2025: Changed file upload area from black to blue-themed styling to match sidebar design