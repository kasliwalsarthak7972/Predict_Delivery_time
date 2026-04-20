import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # ---------------------------------------------------------
    # 1. Load dataset and display basic information
    # ---------------------------------------------------------
    print("--- Step 1: Loading Dataset ---")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("Food_Delivery_Times.csv")
    
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Shape:", df.shape)
    
    print("\nDataset Info:")
    df.info()
    
    # ---------------------------------------------------------
    # 2. Perform data preprocessing
    # ---------------------------------------------------------
    print("\n--- Step 2: Data Preprocessing ---")
    print("\nMissing values before handling:")
    print(df.isnull().sum())
    
    # a) Handle missing values
    # For numerical features, fill missing spots with the median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # For categorical features, fill missing spots with the mode (most frequent value)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    
    # b) Encode categorical features
    # Convert text columns (like Weather, Traffic_Level) into numerical format using One-Hot Encoding
    # drop_first=True prevents the dummy variable trap
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # ---------------------------------------------------------
    # 3. Perform Exploratory Data Analysis (EDA)
    # ---------------------------------------------------------
    print("\n--- Step 3: Exploratory Data Analysis (EDA) ---")
    print("\nSummary Statistics:")
    print(df_encoded.describe())
    
    # Plot 1: Histogram of Delivery Time
    # Helps us understand the distribution of our target variable
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Delivery_Time_min'], kde=True, color='blue')
    plt.title('Distribution of Delivery Time')
    plt.xlabel('Delivery Time (minutes)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Boxplot of Delivery Time by Traffic Level
    # Shows how traffic affects the delivery time and identifies outliers
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Traffic_Level', y='Delivery_Time_min', data=df)
    plt.title('Delivery Time vs Traffic Level')
    plt.xlabel('Traffic Level')
    plt.ylabel('Delivery Time (minutes)')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Scatter Plot of Distance vs Delivery Time
    # Helps us see the linear relationship between distance and time
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Distance_km', y='Delivery_Time_min', data=df, alpha=0.6)
    plt.title('Distance vs Delivery Time')
    plt.xlabel('Distance (km)')
    plt.ylabel('Delivery Time (minutes)')
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # 4. Define features (X) and target (delivery_time)
    # ---------------------------------------------------------
    print("\n--- Step 4: Define Features and Target ---")
    # We drop 'Order_ID' because it's just an ID and has no predictive power
    # We drop 'Delivery_Time_min' because it's our target variable (what we want to predict)
    X = df_encoded.drop(['Order_ID', 'Delivery_Time_min'], axis=1)
    y = df_encoded['Delivery_Time_min']
    
    # ---------------------------------------------------------
    # 5. Split data into training and testing sets
    # ---------------------------------------------------------
    print("\n--- Step 5: Splitting Data ---")
    # 80% data for training, 20% data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # ---------------------------------------------------------
    # 6. Train a Linear Regression model
    # ---------------------------------------------------------
    print("\n--- Step 6: Training Model ---")
    model = LinearRegression()
    model.fit(X_train, y_train) # Learning patterns from the training data
    print("Model training complete.")
    
    # ---------------------------------------------------------
    # 7. Evaluate model
    # ---------------------------------------------------------
    print("\n--- Step 7: Evaluating Model ---")
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # R2 Score: How well the independent variables explain the variance in the target
    r2 = r2_score(y_test, y_pred)
    
    # RMSE: Average error in our predictions (in minutes)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f} minutes")
    
    # ---------------------------------------------------------
    # 8. Plot Actual vs Predicted delivery time
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color='green')
    # Ideal line where Actual == Predicted
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
    plt.title('Actual vs Predicted Delivery Time')
    plt.xlabel('Actual Delivery Time (min)')
    plt.ylabel('Predicted Delivery Time (min)')
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # 9. Print important insights
    # ---------------------------------------------------------
    print("\n--- Step 9: Important Insights ---")
    # Create a DataFrame to view the feature names and their corresponding weights (coefficients)
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    
    # Sort features by the absolute value of their coefficient
    coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False)
    
    print("Top features affecting delivery time most:")
    print(coefficients[['Feature', 'Coefficient']].head(5).to_string(index=False))
    
    most_important_feature = coefficients.iloc[0]['Feature']
    print(f"\nConclusion for Viva: The feature '{most_important_feature}' has the highest impact on the delivery time.")
    print("A positive coefficient means as the feature increases, delivery time increases.")
    print("A negative coefficient means as the feature increases, delivery time decreases.")

if __name__ == "__main__":
    main()
