import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import traceback

# Check if data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

def create_encoder_imputer():
    """
    Create encoder.pkl and imputer.pkl files for the application
    """
    print("Creating encoder and imputer files...")
    
    # Check if the CSV file exists
    csv_path = 'data/traffic_volume.csv'
    if not os.path.exists(csv_path):
        # Try alternative path
        csv_path = 'traffic volume.csv'
        if not os.path.exists(csv_path):
            print(f"Traffic volume CSV file not found at {csv_path}. Please run download_data.py first.")
            return None
    
    print(f"Loading dataset from {csv_path}")
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create a copy of the dataframe
        data = df.copy()
        
        # Check if required columns exist
        required_cols = ['holiday', 'weather_main', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print("Creating dummy encoder and imputer...")
            
            # Create dummy encoder (using sparse_output instead of sparse for newer scikit-learn versions)
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(np.array([['None'], ['Columbus Day']]))
                print("Encoder created successfully")
            except Exception as e:
                print(f"Error creating encoder: {e}")
                print(traceback.format_exc())
                # Try with sparse parameter for older scikit-learn versions
                try:
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoder.fit(np.array([['None'], ['Columbus Day']]))
                    print("Encoder created successfully with sparse=False")
                except Exception as e2:
                    print(f"Error creating encoder with sparse=False: {e2}")
                    print(traceback.format_exc())
                    # Create a simple dummy encoder
                    encoder = "dummy_encoder"
            
            # Create dummy imputer
            try:
                imputer = SimpleImputer(strategy='mean')
                imputer.fit(np.array([[0.0, 0.0, 0.0, 0.0]]))
                print("Imputer created successfully")
            except Exception as e:
                print(f"Error creating imputer: {e}")
                print(traceback.format_exc())
                # Create a simple dummy imputer
                imputer = "dummy_imputer"
        else:
            print("Creating encoder for categorical variables...")
            categorical_cols = ['holiday', 'weather_main']
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(data[categorical_cols])
            
            print("Creating imputer for numerical variables...")
            numerical_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(data[numerical_cols])
        
        # Save encoder and imputer
        print("Saving encoder.pkl...")
        try:
            with open('encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)
            print("encoder.pkl saved successfully")
        except Exception as e:
            print(f"Error saving encoder.pkl: {e}")
            print(traceback.format_exc())
        
        print("Saving imputer.pkl...")
        try:
            with open('imputer.pkl', 'wb') as f:
                pickle.dump(imputer, f)
            print("imputer.pkl saved successfully")
        except Exception as e:
            print(f"Error saving imputer.pkl: {e}")
            print(traceback.format_exc())
        
        print("Encoder and imputer files creation process completed!")
        
        return encoder, imputer
    
    except Exception as e:
        print(f"Error processing the dataset: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    create_encoder_imputer() 