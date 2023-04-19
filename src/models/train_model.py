from src.data import make_dataset
from src.models import evaluate_model
from sklearn.linear_model import LinearRegression

import logging
import pandas as pd
import joblib

# initialize
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

data_train_path = "../../data/processed/lpd_train.csv"
model_save_path = "../../models/MLR.pkl"

# split & load dataset
make_dataset.make()
df = pd.read_csv(data_train_path)

# declare and train a Linear Regression model
mlr = LinearRegression()

X_train = df[['Principal', 'terms', 'past_due_days',
              'age', 'education', 'Gender']]
y_train = df[['loan_status']]

logging.info('Training the model...')
mlr.fit(X_train, y_train)

accuracy = mlr.score(X_train, y_train)
print('Train score: ', mlr.score(X_train, y_train))

# save the model
joblib.dump(mlr, model_save_path)

# evaluate the model
evaluate_model.run()
