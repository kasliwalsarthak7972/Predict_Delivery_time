import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Setup Page Configuration
st.set_page_config(page_title="PredictDelivery AI", page_icon="🛵", layout="wide")

# Custom CSS for better styling (matching our modern theme)
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1, h2, h3 {
        color: #ff6b35 !important;
    }
    .stButton>button {
        background-color: #ff6b35;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #ff8559;
        color: white;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-time {
        font-size: 48px;
        font-weight: bold;
        color: #ff6b35;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train_model():
    """Loads dataset, preprocesses, and trains the model. Cached for performance."""
    df = pd.read_csv("Food_Delivery_Times.csv")
    
    # Preprocessing
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop(['Order_ID', 'Delivery_Time_min'], axis=1)
    y = df_encoded['Delivery_Time_min']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, X.columns.tolist(), df


# Initialize model and data
model, training_columns, raw_df = load_and_train_model()

st.title("🛵 PredictDelivery AI")
st.write("Smart, AI-powered food delivery time estimation based on real-time conditions.")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Order Details")
    
    distance = st.number_input("Distance (km)", min_value=0.0, value=5.2, step=0.1)
    prep_time = st.number_input("Preparation Time (min)", min_value=0.0, value=15.0, step=1.0)
    experience = st.number_input("Courier Experience (Years)", min_value=0.0, value=2.5, step=0.5)
    
    vehicle = st.selectbox("Vehicle Type", ["Scooter", "Bike", "Car"])
    weather = st.selectbox("Weather", ["Clear", "Windy", "Foggy", "Rainy", "Snowy"])
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    
    predict_clicked = st.button("Estimate Time")

with col2:
    if predict_clicked:
        # Create input dictionary
        input_data = {
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [experience],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Ensure identical one-hot encoding
        cat_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        input_encoded = pd.get_dummies(input_df, columns=cat_cols)
        
        aligned_df = pd.DataFrame(0, index=np.arange(1), columns=training_columns)
        
        for col in input_encoded.columns:
            if col in aligned_df.columns:
                aligned_df.at[0, col] = int(input_encoded[col].iloc[0])
                
        # Predict
        prediction = model.predict(aligned_df)
        predicted_time = max(0, int(round(prediction[0])))
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2>Estimated Delivery Time</h2>
                <div class="prediction-time">{predicted_time} <span style="font-size: 24px; color: #f8fafc;">Minutes</span></div>
                <p style="color: #94a3b8;">Based on current conditions and historical data.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="prediction-box" style="opacity: 0.5;">
                <h2>Estimated Delivery Time</h2>
                <div class="prediction-time">-- <span style="font-size: 24px; color: #f8fafc;">Minutes</span></div>
                <p style="color: #94a3b8;">Enter details and click Estimate Time</p>
            </div>
        """, unsafe_allow_html=True)

