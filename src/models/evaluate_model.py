import joblib
import numpy as np
import json

import pandas as pd


def load_config():
    with open('../config.json') as config_file:
        config = json.load(config_file)
        return config


def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy


def init():
    cfg = load_config()
    project_dir = cfg["project_dir"]
    model_save_path = project_dir + "models/MLR.pkl"
    data_test_path = project_dir + "data/processed/lpd_test.csv"
    df = pd.read_csv(data_test_path)

    # load the model
    mlr = joblib.load(model_save_path)
    X_test = df[['Principal', 'terms', 'past_due_days',
                 'age', 'education', 'Gender']]
    y_test = df[['loan_status']]
    evaluate_model(mlr, X_test, y_test)
