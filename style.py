import streamlit as st

def apply_glassy_style():
    """
    Apply glassy style to the Streamlit app
    """
    glassy_css = """
    <style>
        /* Glassy look CSS */
        .stApp {
            background: linear-gradient(135deg, rgba(10, 10, 30, 0.9), rgba(10, 20, 40, 0.9)), 
                        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            background-size: cover;
            background-attachment: fixed;
        }
        
        div.css-1r6slb0.e1tzin5v2 {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        div.stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        div.stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px 4px 0px 0px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            padding: 8px 16px;
            transition: all 0.3s ease;
        }
        
        div.stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-bottom: 2px solid #ff4b4b;
        }
        
        /* Cards design */
        .glass-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px 0 rgba(31, 38, 135, 0.47);
        }
        
        /* Button styling */
        div.stButton button {
            background: rgba(255, 75, 75, 0.7);
            border: none;
            border-radius: 5px;
            color: white;
            transition: all 0.3s ease;
        }
        
        div.stButton button:hover {
            background: rgba(255, 75, 75, 0.9);
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(255, 75, 75, 0.3);
        }
        
        /* File uploader styling */
        div.stFileUploader {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }
        
        /* Progress bar styling */
        div.stProgress > div > div {
            background-color: #ff4b4b;
        }
        
        /* Metric containers */
        div.css-1ht1j8u.e16fv1kl0 {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* Headers styling */
        h1, h2, h3 {
            color: white;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Dataframe styling */
        div.stDataFrame div[data-testid="stTable"] {
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
    </style>
    """
    st.markdown(glassy_css, unsafe_allow_html=True)

def render_header():
    """
    Render the application header with logo and title
    """
    header_html = """
    <div style="display: flex; align-items: center; margin-bottom: 20px; background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; backdrop-filter: blur(10px);">
        <div style="margin-right: 20px;">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="24" height="24" rx="5" fill="rgba(255,75,75,0.8)"/>
                <path d="M5 9C5 7.89543 5.89543 7 7 7H17C18.1046 7 19 7.89543 19 9V15C19 16.1046 18.1046 17 17 17H7C5.89543 17 5 16.1046 5 15V9Z" stroke="white" stroke-width="1.5"/>
                <path d="M7.5 10.5H16.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
                <path d="M7.5 13.5H12.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
        </div>
        <div>
            <h1 style="margin:0; color: white; font-size: 2.2em; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">Credit Card Fraud Detection</h1>
            <p style="margin:0; color: rgba(255,255,255,0.8); font-size: 1.1em;">Advanced Machine Learning Protection System</p>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_glass_card(title, content, key=None):
    """
    Render content inside a glass-styled card
    
    Parameters:
    -----------
    title : str
        Title of the card
    content : str
        HTML content inside the card
    key : str, optional
        Unique key for the card
    """
    st.markdown(f"""
    <div class="glass-card" id="{key if key else ''}">
        <h3 style="margin-top:0;">{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

def render_model_card(model_name, description, accuracy, icon):
    """
    Render a model information card
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    description : str
        Description of the model
    accuracy : float
        Accuracy of the model
    icon : str
        SVG icon for the model
    """
    card_html = f"""
    <div class="glass-card">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="margin-right: 15px;">
                {icon}
            </div>
            <div>
                <h3 style="margin: 0;">{model_name}</h3>
                <div style="display: flex; align-items: center;">
                    <div style="background: linear-gradient(90deg, rgba(255,75,75,0.7) {accuracy*100}%, rgba(255,255,255,0.1) {accuracy*100}%); 
                                height: 6px; width: 100px; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="color: white;">{accuracy:.2%}</span>
                </div>
            </div>
        </div>
        <p style="margin: 0; color: rgba(255,255,255,0.8);">{description}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_feature_importance(feature_names, importances):
    """
    Render feature importance visualization
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : list
        List of importance values
    """
    # Create a horizontal bar chart
    html = """
    <div style="margin-top: 20px;">
        <h3>Feature Importance</h3>
        <div style="overflow-y: auto; max-height: 300px;">
    """
    
    # Sort features by importance
    sorted_indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
    max_importance = max(importances) if importances else 1
    
    for i in sorted_indices:
        percentage = (importances[i] / max_importance) * 100
        html += f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: rgba(255,255,255,0.9);">{feature_names[i]}</span>
                <span style="color: rgba(255,255,255,0.7);">{importances[i]:.4f}</span>
            </div>
            <div style="background: rgba(255,255,255,0.1); border-radius: 5px; height: 8px; width: 100%;">
                <div style="background: linear-gradient(90deg, rgba(255,75,75,0.8), rgba(255,150,75,0.8)); 
                            width: {percentage}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
