import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Check if data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Check if models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def load_data():
    """
    Load the traffic volume dataset
    """
    # Check if the CSV file exists
    if not os.path.exists('data/traffic_volume.csv'):
        print("Traffic volume CSV file not found. Please run download_data.py first.")
        return None
    
    # Load the dataset
    df = pd.read_csv('data/traffic_volume.csv')
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess the traffic volume dataset
    """
    print("Preprocessing data...")
    
    # Make a copy of the dataframe
    data = df.copy()
    
    # Convert date_time to datetime
    data['date_time'] = pd.to_datetime(data['date_time'])
    
    # Extract date and time features
    data['year'] = data['date_time'].dt.year
    data['month'] = data['date_time'].dt.month
    data['day'] = data['date_time'].dt.day
    data['hour'] = data['date_time'].dt.hour
    data['minute'] = data['date_time'].dt.minute
    data['second'] = data['date_time'].dt.second
    data['dayofweek'] = data['date_time'].dt.dayofweek
    
    # Convert holiday to numeric
    holiday_mapping = {
        'None': 0,
        'Columbus Day': 1,
        'Veterans Day': 2,
        'Thanksgiving Day': 3,
        'Christmas Day': 4,
        'New Years Day': 5,
        'Washingtons Birthday': 6,
        'Memorial Day': 7,
        'Independence Day': 8,
        'State Fair': 9,
        'Labor Day': 10,
        'Martin Luther King Jr Day': 11
    }
    
    data['holiday_numeric'] = data['holiday'].map(holiday_mapping)
    
    # Convert weather_main to numeric
    weather_mapping = {
        'Clear': 0,
        'Clouds': 1,
        'Fog': 2,
        'Drizzle': 3,
        'Rain': 4,
        'Mist': 5,
        'Haze': 6,
        'Smoke': 7,
        'Snow': 8,
        'Squall': 9,
        'Thunderstorm': 10
    }
    
    data['weather_numeric'] = data['weather_main'].map(weather_mapping)
    
    # Create binary features for rain and snow
    data['rain'] = data['rain_1h'].apply(lambda x: 1 if x > 0 else 0)
    data['snow'] = data['snow_1h'].apply(lambda x: 1 if x > 0 else 0)
    
    # Select features for the model
    features = ['holiday_numeric', 'temp', 'rain', 'snow', 'weather_numeric', 
                'year', 'month', 'day', 'hour', 'minute', 'second']
    
    # Rename columns to match the form input names
    column_mapping = {
        'holiday_numeric': 'holiday',
        'weather_numeric': 'weather',
        'hour': 'hours',
        'minute': 'minutes',
        'second': 'seconds'
    }
    
    data = data.rename(columns=column_mapping)
    
    # Select features and target
    X = data[['holiday', 'temp', 'rain', 'snow', 'weather', 
              'year', 'month', 'day', 'hours', 'minutes', 'seconds']]
    y = data['traffic_volume']
    
    return X, y

def train_model(X, y):
    """
    Train a Random Forest model on the traffic volume data
    """
    print("Training model...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Save the model and scaler
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(scaler, open('scale.pkl', 'wb'))
    
    print("Model and scaler saved as model.pkl and scale.pkl")
    
    # Create a directory for visualizations
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Traffic Volume')
    plt.tight_layout()
    plt.savefig('static/actual_vs_predicted.png')
    
    print("Visualizations saved in the static directory")
    
    return model, scaler

if __name__ == "__main__":
    # Load the data
    df = load_data()
    
    if df is not None:
        # Preprocess the data
        X, y = preprocess_data(df)
        
        # Train the model
        model, scaler = train_model(X, y) 