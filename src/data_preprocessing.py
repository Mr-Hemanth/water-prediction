import pandas as pd

def preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Handle missing values (example: fill with mean for numerical columns)
    data.fillna(data.mean(), inplace=True)

    # Encode categorical variables if any (example using one-hot encoding)
    data = pd.get_dummies(data)

    # Define features (X) and target (y)
    X = data.drop('target', axis=1)  # Replace 'target' with the actual target column name
    y = data['target']  # Ensure 'target' matches the column name in your dataset

    return X, y
