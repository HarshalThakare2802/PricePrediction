# Real Estate Price Prediction

This project is a comprehensive implementation of a machine learning-based application to predict real estate prices in Nashik City. It uses BERT for processing textual data, specifically the location of the property, and numerical preprocessing techniques for other features. The web interface, built with Flask, enables users to input property details and receive an estimated price.

## Table of Contents

1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Project Workflow](#project-workflow)
4. [Dataset](#dataset)
5. [Model Training](#model-training)
6. [Web Application](#web-application)
7. [Installation and Usage](#installation-and-usage)
8. [File Structure](#file-structure)

Overview

The project predicts real estate prices based on multiple features including:
- Location
- Bedrooms
- Bathrooms
- Living area in square feet
- Lot size
- Number of floors
- Year built and renovated

It leverages a pre-trained BERT model to handle textual data (location) and a linear regression model for price prediction. The application provides a web-based interface for easy interaction.

# Technologies Used

- Python
- Flask (Web framework)
- Scikit-learn (Machine learning library)
- Hugging Face Transformers (for BERT-based feature extraction)
- Pandas and NumPy (Data manipulation and processing)
- Joblib (Model serialization)
- HTML (Frontend)

# Project Workflow

1. Data Preprocessing:
   - Handled categorical (location) and numerical features.
   - Standardized numerical data using `StandardScaler`.
   - Encoded categorical data using `OneHotEncoder`.

2. Feature Engineering:
   - Extracted textual features from `Location` using a pre-trained BERT model.
   - Combined numerical and text features for training.

3. Model Training:
   - Used linear regression to predict real estate prices.
   - Saved the trained model and preprocessor for deployment.

4. Web Application:
   - Built a Flask app to accept user inputs and predict prices.
   - Rendered predictions in a user-friendly HTML interface.

# Dataset

The dataset includes the following features:
- Location
- Bedrooms
- Bathrooms
- Square footage (living and lot area)
- Number of floors
- Year built and renovated
- Price (target variable)

The dataset file is named `NashikCity_house_data.csv`.

# Model Training

- The training process is executed in `pricepredication.py`.
- Key steps include:
  - Data loading and preprocessing.
  - Feature extraction using BERT for `Location`.
  - Combining numerical and text-based features.
  - Training a linear regression model.
  - Saving the trained model and preprocessor as `.pkl` files.

# Web Application

- The Flask app (`app2.py`) serves as the interface for users to input data and get predictions.
- Features:
  - Input fields for all required property details.
  - A "Predict" button to compute the estimated price.
  - Error handling and user feedback.

# Installation and Usage

# Prerequisites

1. Python 3.8 or above.
2. Required Python libraries:
   - Flask
   - Scikit-learn
   - Transformers
   - Pandas
   - NumPy
   - Joblib
   - PyTorch

Install dependencies using:
```bash
pip install -r requirements.txt
```

# Steps to Run the Project

1. Train the Model:
   - Run `pricepredication.py` to train the model and generate the `.pkl` files.
   ```bash
   python pricepredication.py
   ```

2. Start the Flask App:
   - Run `app2.py` to launch the web application.
   ```bash
   python app2.py
   ```

3. Access the Application:
   - Open a browser and navigate to `http://127.0.0.1:5000/`.

4. Predict Real Estate Prices:
   - Enter the property details in the form and click "Predict" to get the estimated price.

# File Structure

```plaintext
├── NashikCity_house_data.csv        # Dataset
├── pricepredication.py             # Model training script
├── app2.py                         # Flask application
├── templates/
│   └── index.html                  # HTML template for the web app
├── real_estate_model.pkl           # Serialized model
├── real_estate_model_with_preprocessor.pkl # Serialized model with preprocessor
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

# Example Predictions

The trained model provides estimated prices for real estate properties based on the provided features. Example predictions are printed during the training process in `pricepredication.py`.

# Future Enhancements

1. Include additional features such as proximity to landmarks or market trends.
2. Enhance the model with advanced regression techniques or neural networks.
3. Improve the web interface with better styling and input validation.

---


