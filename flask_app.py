import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global variables for model and training columns
model = None
training_columns = None

def train_model():
    """Loads data, preprocesses it, and trains the model upon startup."""
    global model, training_columns
    print("Training model...")
    
    csv_path = "Food_Delivery_Times.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Handle missing values (same as in delivery_prediction.py)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # 2. One-hot encode
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 3. Define X and y
    X = df_encoded.drop(['Order_ID', 'Delivery_Time_min'], axis=1)
    y = df_encoded['Delivery_Time_min']
    
    # Save training columns for aligning incoming predict requests
    training_columns = X.columns.tolist()
    
    # 4. Train model
    model = LinearRegression()
    model.fit(X, y) # Training on full dataset for the app
    print("Model trained successfully. Ready to serve predictions!")

# Train model before the first request
train_model()

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or training_columns is None:
        return jsonify({'error': 'Model is not trained yet. Check server logs.'}), 500
        
    data = request.json
    
    try:
        # Create a DataFrame from the input data
        # Data should contain: Distance_km, Preparation_Time_min, Courier_Experience_yrs, Weather, Traffic_Level, Time_of_Day, Vehicle_Type
        input_data = {
            'Distance_km': [float(data.get('Distance_km', 0))],
            'Preparation_Time_min': [float(data.get('Preparation_Time_min', 0))],
            'Courier_Experience_yrs': [float(data.get('Courier_Experience_yrs', 0))],
            'Weather': [data.get('Weather', 'Clear')],
            'Traffic_Level': [data.get('Traffic_Level', 'Low')],
            'Time_of_Day': [data.get('Time_of_Day', 'Afternoon')],
            'Vehicle_Type': [data.get('Vehicle_Type', 'Scooter')]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # We need to one-hot encode the input identically to the training data.
        # So we create dummies, but we must ensure it has the exact same columns as the training set
        cat_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)
        
        # Align with training columns (fill missing with 0)
        # We start with a dataframe of zeros with the exact same shape as training columns
        aligned_df = pd.DataFrame(0, index=np.arange(1), columns=training_columns)
        
        # For each column in input_encoded, if it exists in aligned_df, we set the value
        for col in input_encoded.columns:
            if col in aligned_df.columns:
                # pandas get_dummies returns boolean, cast to int
                val = input_encoded[col].iloc[0]
                aligned_df.at[0, col] = int(val) if isinstance(val, (bool, np.bool_)) else val
                
            # Note: For drop_first=True, some categories are dropped (e.g. Weather_Clear). 
            # If the user selects Weather=Clear, it simply means all other Weather dummy variables are 0.
            # This is naturally handled because our aligned_df is initialized with 0s.
        
        # Make prediction
        prediction = model.predict(aligned_df)
        
        # Ensure we don't predict negative time
        predicted_time = max(0, int(round(prediction[0])))
        
        return jsonify({'predicted_time_min': predicted_time})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
