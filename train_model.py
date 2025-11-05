import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging

logging.basicConfig(level=logging.INFO)

# Load and prepare the data
df = pd.read_csv('Crop_recommendation.csv')
logging.info(f"Unique crops in dataset: {df['label'].unique()}")

# Prepare features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Create label mapping
unique_crops = sorted(df['label'].unique())
crop_to_num = {crop: i+1 for i, crop in enumerate(unique_crops)}
num_to_crop = {i+1: crop for i, crop in enumerate(unique_crops)}

# Convert labels to numbers
y = y.map(crop_to_num)

logging.info("Crop mapping:")
for num, crop in num_to_crop.items():
    logging.info(f"{num}: {crop}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
ms = MinMaxScaler()
sc = StandardScaler()

X_train_minmax = ms.fit_transform(X_train)
X_train_scaled = sc.fit_transform(X_train_minmax)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the scalers and model
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

# Test prediction
test_data = np.array([[90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362]])
test_minmax = ms.transform(test_data)
test_scaled = sc.transform(test_minmax)
prediction = model.predict(test_scaled)
logging.info(f"Test prediction for rice sample: {prediction[0]} -> {num_to_crop[prediction[0]]}")

# Save the mapping
with open('crop_mapping.txt', 'w') as f:
    for num, crop in num_to_crop.items():
        f.write(f"{num}: {crop}\n")

logging.info("Model, scalers, and mapping saved successfully")