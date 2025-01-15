import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv(r'C:\Users\230156\Downloads\NashikCity_house_data.csv')






# # Handle missing values
# data = data.fillna(data.mean())


# Separate features and target
X = data[['Location', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
          'floors', 'sqft_above', 'sqft_basement', 'yr_built',
          'yr_renovated', 'sqft_living15', 'sqft_lot15']]
y = data['Price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Process numerical and categorical features
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                      'floors', 'sqft_above', 'sqft_basement', 'yr_built',
                      'yr_renovated', 'sqft_living15', 'sqft_lot15']
categorical_features = ['Location']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Initialize text tokenizer and model for location-based feature extraction
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

def extract_text_features(texts):
    """Process textual data using a pre-trained transformer."""
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output

# Process location features
text_features_train = torch.cat([extract_text_features([loc]) for loc in X_train['Location']])
text_features_test = torch.cat([extract_text_features([loc]) for loc in X_test['Location']])

# Transform numerical and categorical features
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Combine text features with other features
X_train_combined = torch.cat((torch.tensor(X_train_transformed.toarray(), dtype=torch.float32), text_features_train), dim=1)
X_test_combined = torch.cat((torch.tensor(X_test_transformed.toarray(), dtype=torch.float32), text_features_test), dim=1)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train_combined.numpy(), y_train)

# Evaluate the model
y_pred = regressor.predict(X_test_combined.numpy())
print("Model Trained. Example Predictions:")
print(y_pred[:5])

# Save both the model and preprocessor together
joblib.dump({'model': regressor, 'preprocessor': preprocessor}, 'real_estate_model_with_preprocessor.pkl')
print("Model and Preprocessor saved as 'real_estate_model_with_preprocessor.pkl'")

# Save the model separately if you still want to keep it
joblib.dump(regressor, 'real_estate_model.pkl')
print("Model saved as 'real_estate_model.pkl'")

# Dataset preview (optional)
print("Dataset Head:\n", data.head())





