from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
# Only load from the 'real_estate_model_with_preprocessor.pkl' file
model_data = joblib.load('real_estate_model_with_preprocessor.pkl')
model = model_data['model']
preprocessor = model_data['preprocessor']

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Define a function to preprocess location input using BERT
def extract_text_features(location):
    inputs = tokenizer(location, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output

@app.route('/')
def home():
    return render_template('index.html')  # The homepage (HTML form)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        form_data = {
            'Location': request.form['Location'],
            'Bedrooms': request.form['Bedrooms'],
            'Bathrooms': request.form['Bathrooms'],
            'Sqft_Living': request.form['Sqft_Living'],
            'Sqft_Lot': request.form['Sqft_Lot'],
            'Floors': request.form['Floors'],
            'Sqft_Above': request.form['Sqft_Above'],
            'Sqft_Basement': request.form['Sqft_Basement'],
            'Yr_Built': request.form['Yr_Built'],
            'Yr_Renovated': request.form['Yr_Renovated'],
            'Sqft_Living15': request.form['Sqft_Living15'],
            'Sqft_Lot15': request.form['Sqft_Lot15'],
        }

        # Preprocess and predict
        location = form_data['Location']
        bedrooms = int(form_data['Bedrooms'])
        bathrooms = int(form_data['Bathrooms'])
        sqft_living = float(form_data['Sqft_Living'])
        sqft_lot = float(form_data['Sqft_Lot'])
        floors = float(form_data['Floors'])
        sqft_above = float(form_data['Sqft_Above'])
        sqft_basement = float(form_data['Sqft_Basement'])
        yr_built = int(form_data['Yr_Built'])
        yr_renovated = int(form_data['Yr_Renovated'])
        sqft_living15 = float(form_data['Sqft_Living15'])
        sqft_lot15 = float(form_data['Sqft_Lot15'])

        input_data = pd.DataFrame([[location, bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                                    sqft_above, sqft_basement, yr_built, yr_renovated,
                                    sqft_living15, sqft_lot15]],
                                  columns=['Location', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                                           'floors', 'sqft_above', 'sqft_basement', 'yr_built',
                                           'yr_renovated', 'sqft_living15', 'sqft_lot15'])
        processed_data = preprocessor.transform(input_data)

        # Process location feature
        location_features = extract_text_features(location)

        # Combine numerical and text features
        numerical_tensor = torch.tensor(processed_data.toarray(), dtype=torch.float32) if hasattr(processed_data, 'toarray') else torch.tensor(processed_data, dtype=torch.float32)
        combined_features = torch.cat((numerical_tensor, location_features), dim=1)

        # Make a prediction
        prediction = model.predict(combined_features.numpy())[0]

        return render_template('index.html', prediction_text=f"Estimated Price: ₹{round(prediction, 2)}")

    except Exception as e:
        return render_template('index.html', prediction_text="Error occurred: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
