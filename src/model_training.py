import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from collections import Counter

# Load and preprocess the data
def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Check for missing values and handle them
    if data.isnull().sum().sum() > 0:
        data.fillna(data.mean(), inplace=True)  # Fill missing values with column means
    
    # Assuming your target variable is named 'target', and it contains binary labels (0 and 1)
    # You can modify this to fit your dataset
    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target variable
    
    print(f"Class distribution:\n{Counter(y)}")
    
    return X, y

# Main code
file_path = '/Users/hemanth/Desktop/WaterQualityPrediction/data/water_quality_data.csv'  # <-- Update with your actual CSV path
X, y = preprocess_data(file_path)

# Split the data
test_size = 0.2
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
except ValueError as e:
    print(f"Error during train-test split: {e}")
    # Handle small dataset class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional, but useful for debugging)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Save the trained model to a .pkl file
joblib.dump(model, 'logistic_regression_model.pkl')
print("Model saved as 'logistic_regression_model.pkl'")

# Save the scaler to a file (important for preprocessing new input data)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")
