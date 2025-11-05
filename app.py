from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load trained model and scalers (raise clear errors if missing)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('standscaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    with open('minmaxscaler.pkl', 'rb') as f:
        ms = pickle.load(f)
    # Load crop mapping from file
    crop_dict = {}
    with open('crop_mapping.txt', 'r') as f:
        for line in f:
            num, crop = line.strip().split(': ')
            crop_dict[int(num)] = crop
    logging.info(f"Loaded crop mapping: {crop_dict}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required file not found: {e}")
except Exception as e:
    raise Exception(f"Error loading model or scalers: {str(e)}")

# Flask app setup (matches actual folder `Static`)
app = Flask(__name__, static_folder='Static')

# Test prediction with known data
test_data = np.array([[90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362]])
logging.info("Testing model with known rice data...")
try:
    test_scaled = ms.transform(test_data)
    test_final = sc.transform(test_scaled)
    test_pred = model.predict(test_final)
    logging.info(f"Test prediction result: {test_pred}")
except Exception as e:
    logging.error(f"Test prediction failed: {e}")

@app.route('/')
def index():
    return render_template("index.html")


# Map plural/singular variations to correct image filenames
def get_image_filename(crop_name: str) -> str:
    """Get the correct image filename for a crop name, handling plural/singular variations."""
    # Dictionary of special cases and their corresponding image names
    special_cases = {
        "chickpea": "Images/chickpeas.png",
        "chickpeas": "Images/chickpeas.png",
        "pigeonpea": "Images/pigeonpeas.png",
        "pigeonpeas": "Images/pigeonpeas.png",
        "kidneybean": "Images/kidneybeans.png",
        "kidneybeans": "Images/kidneybeans.png",
        "mothbean": "Images/mothbeans.png",
        "mothbeans": "Images/mothbeans.png",
        "mungbean": "Images/mungbean.png",
    }
    
    # Convert to lowercase for case-insensitive matching
    crop_lower = crop_name.lower()
    
    # Check special cases first
    if crop_lower in special_cases:
        return special_cases[crop_lower]
    
    # For standard cases, use Images folder with proper casing
    return f"Images/{crop_lower}.png"

def resolve_image_filename(app_obj, crop_name: str) -> str:
    """Get the correct image filename for a crop, with fallback to default.jpg"""
    image_file = get_image_filename(crop_name)
    image_path = os.path.join(app_obj.static_folder, image_file)
    
    if os.path.exists(image_path):
        logging.info(f"Found image for crop {crop_name}: {image_file}")
        return image_file
    
    logging.warning(f"No image found for crop {crop_name} at {image_path}, using default")
    return "Images/default.jpg"


@app.route('/predict', methods=['POST'])
def predict():
    # Safely parse and validate inputs
    try:
        N = float(request.form.get('Nitrogen', ''))
        P = float(request.form.get('Phosporus', ''))
        K = float(request.form.get('Potassium', ''))
        temp = float(request.form.get('Temperature', ''))
        humidity = float(request.form.get('Humidity', ''))
        ph = float(request.form.get('Ph', ''))
        rainfall = float(request.form.get('Rainfall', ''))
    except (ValueError, TypeError):
        result = "Invalid input — please enter numeric values for all fields."
        return render_template('index.html', result=result, crop_image='images/default.jpg')

    # Input validation based on training data ranges
    if not (20 <= temp <= 30 or 5 <= ph <= 8 or 60 <= humidity <= 100):
        result = "Input values out of expected range. Please check your values."
        return render_template('index.html', result=result, crop_image='images/default.jpg')

    # Prepare features in the correct order matching training data
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    logging.info(f"Input values: N={N}, P={P}, K={K}, temp={temp}, humidity={humidity}, ph={ph}, rainfall={rainfall}")
    single_pred = np.array(feature_list, dtype=float).reshape(1, -1)

    # Apply scalers and predict (capture runtime errors)
    try:
        logging.info("================== NEW PREDICTION ==================")
        logging.info(f"Raw input features: {feature_list}")
        
        # Scale the features in correct order: MinMax first, then Standard
        minmax_scaled = ms.transform(single_pred)
        logging.info(f"After MinMax scaling: {minmax_scaled.tolist()}")
        
        final_features = sc.transform(minmax_scaled)
        logging.info(f"After Standard scaling: {final_features.tolist()}")
        
        # Make prediction
        prediction = model.predict(final_features)
        logging.info(f"Raw prediction: {prediction[0]}")
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(final_features)
            pred_class = prediction[0]
            confidence = probs[0][pred_class - 1]  # -1 because classes are 1-based
            logging.info(f"Confidence: {confidence:.3f}")
            
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        result = f"Prediction error: {str(e)}"
        return render_template('index.html', result=result, crop_image='images/default.jpg')

    pred_value = prediction[0]
    logging.info(f"Prediction value: {pred_value}, type: {type(pred_value)}")

    # Convert numpy numeric types to Python int
    if isinstance(pred_value, (np.integer, np.floating)):
        pred_value = int(pred_value.item())

    # Use the crop mapping loaded from crop_mapping.txt (keys are integers -> lowercase crop names)
    crop = None
    if isinstance(pred_value, (int, float)):
        key = int(pred_value)
        if key in crop_dict:
            crop = crop_dict[key]  # this is the lowercase label saved during training
            logging.info(f"Mapped numeric prediction {key} to crop: {crop}")
        else:
            logging.error(f"Prediction {key} not found in loaded crop mapping")
            crop = str(pred_value)
    else:
        # If model returned a string label, use it directly
        crop = str(pred_value).strip()
        logging.info(f"Using prediction directly as crop name: {crop}")

    # Normalize crop display name (title case) but keep original lowercase for filename resolution
    display_crop = crop.title()

    # Resolve image filename robustly and return template
    image_file = resolve_image_filename(app, crop)
    result = f"{display_crop} is the best crop to be cultivated right there"
    return render_template('index.html', result=result, crop_image=image_file)


if __name__ == '__main__':
    app.run(debug=True)
