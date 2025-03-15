# Traffic Volume Estimation

This project is a web application that predicts traffic volume based on various input parameters such as weather conditions, date, time, and holidays.

## Project Structure

```
traffic-volume-estimation/
│
├── app.py                  # Main Flask application
├── create_model.py         # Script to create placeholder model files
├── download_data.py        # Script to download the traffic volume dataset
├── train_model.py          # Script to train the model on the dataset
├── model.pkl               # Trained machine learning model
├── scale.pkl               # Scaler for data preprocessing
├── requirements.txt        # Python dependencies
│
├── data/                   # Directory for dataset files
│   └── traffic_volume.csv  # Traffic volume dataset
│
├── templates/              # HTML templates
│   ├── index.html          # Main prediction page
│   ├── visualize.html      # Visualization page
│   └── about.html          # About page
│
└── static/                 # Static files (CSS, JS, images)
    ├── feature_importance.png  # Feature importance visualization
    └── actual_vs_predicted.png # Model performance visualization
```

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download and prepare the dataset:
   ```
   python download_data.py
   ```
4. Train the model:
   ```
   python train_model.py
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Fill in the form with the required parameters and click "Predict" to get the estimated traffic volume
4. Navigate to the "View Visualizations" page to see model insights
5. Visit the "About" page to learn more about the project

## Input Parameters

- **Holiday**: Select a holiday if applicable
- **Temperature**: Enter the temperature value
- **Rain**: Enter 1 if it's raining, 0 otherwise
- **Snow**: Enter 1 if it's snowing, 0 otherwise
- **Weather**: Select the weather condition
- **Date and Time**: Enter the year, month, day, hours, minutes, and seconds

## Model

The application uses a Random Forest Regressor model trained on the Metro Interstate Traffic Volume dataset. The model takes into account various factors that affect traffic volume and provides an estimate based on the input parameters.

## Dataset

The model is trained on the Metro Interstate Traffic Volume dataset, which includes hourly traffic volume for Interstate 94 Westbound in Minneapolis-St Paul, Minnesota. The dataset includes weather features and holiday information alongside the traffic volume data.

## Features

- **Prediction**: Enter parameters and get traffic volume predictions
- **Visualizations**: View model insights and performance metrics
- **About**: Learn about the project, dataset, and model
- **Responsive Design**: User-friendly interface that works on various devices 