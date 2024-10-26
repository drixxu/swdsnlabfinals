import pandas as pd 
import numpy as np 
import os 
import joblib 
from sklearn.model_selection import train_test_split 
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Constants
DATASET = 'datasets/preprocessed1.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm.pkl')

# Load the dataset
data = pd.read_csv(DATASET)

# Feature Engineering - SPLIT
features = [
    'YEAR',
    'MONTH',
    'ITEM CODE',
    'RETAIL TRANSFERS',
    'WAREHOUSE SALES',
]
target = 'DEMAND CLASS'

# Train the model
X = np.asarray(data[features])
y = np.asarray(data[target])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model 
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, MODEL_PATH)