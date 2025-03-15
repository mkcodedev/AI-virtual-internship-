import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create a simple model for demonstration
model = RandomForestRegressor(n_estimators=10, random_state=42)

# Create some dummy data
X = np.random.rand(100, 11)  # 11 features as in our form
y = np.random.rand(100) * 5000  # Random traffic volume values

# Fit the model
model.fit(X, y)

# Create a scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the model and scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scale.pkl', 'wb'))

print("Model and scaler files created successfully!") 