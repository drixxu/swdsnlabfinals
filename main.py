import joblib 
import inquirer
from src.inputs import get_user_input 
import warnings
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the models
models = {
    'K-Nearest Neighbor (84%)': 'models/k-nn.pkl',
    'AdaBoost (85%)': 'models/ab.pkl',
    'Gradient Boosting (86%)': 'models/gbm.pkl',
    'CatBoost (86%)': 'models/cb.pkl',
    'LightGBM (86%)': 'models/lgbm.pkl',
    'XGBoost (86%)': 'models/xgb.pkl',
}

# Define the initial question
initial = [
    inquirer.List(
        'action',
        message='What would you like to do?',
        choices=['Predict', 'Exit']
    )
]

# Prompt for the initial action
initial_answer = inquirer.prompt(initial)

# Action 
if initial_answer['action'] == 'Predict':
    # Choose the model
    model_choice = inquirer.List(
        'model',
        message='Which model would you like to use?',
        choices=list(models.keys())
    )
    selected_model = inquirer.prompt([model_choice])['model']
    
    # Load the selected model
    model = joblib.load(models[selected_model])

    # Get user input
    new_data = get_user_input()  # Ensure this returns a DataFrame or 2D array

     # Make a prediction
    prediction = model.predict(new_data)

    # Map Numerical Prediction
    label_map = {
    0.00: 'Low',
    1.00: 'Medium',
    2.00: 'High'
    }

    # Check if prediction is an array
    if isinstance(prediction, np.ndarray):
        # Map each element in prediction array
        for i, pred in enumerate(prediction):
            prediction_label = label_map.get(float(pred), 'Unknown')
            print(f"Prediction {i + 1}: {prediction_label} Demand")
    else:
        # Map single prediction
        prediction_label = label_map.get(float(prediction), 'Unknown')
        print(f"Prediction: {prediction_label} Demand")


else:
    print('Exit, mama mo bye.')

