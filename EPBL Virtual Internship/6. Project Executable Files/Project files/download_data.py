import os
import pandas as pd
import requests
from io import StringIO

# URL for the Metro Interstate Traffic Volume dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"

def download_data():
    """
    Download the traffic volume dataset from UCI Machine Learning Repository
    """
    print("Downloading traffic volume dataset...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download the file
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the compressed file
        with open('data/traffic_volume.csv.gz', 'wb') as f:
            f.write(response.content)
        
        print("Download completed. File saved as data/traffic_volume.csv.gz")
        
        # Read and save as CSV
        df = pd.read_csv('data/traffic_volume.csv.gz', compression='gzip')
        df.to_csv('data/traffic_volume.csv', index=False)
        
        print("CSV file extracted and saved as data/traffic_volume.csv")
        print(f"Dataset shape: {df.shape}")
        print("\nSample data:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        return None

if __name__ == "__main__":
    download_data() 