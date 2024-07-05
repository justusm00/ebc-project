from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle


from columns import COLS_FEATURES, COLS_LABELS, COLS_DAYOFYEAR
from paths import PATH_MLP_TRAINING

training_data = pd.read_csv(PATH_MLP_TRAINING + 'training_data.csv')
test_data = pd.read_csv(PATH_MLP_TRAINING + 'test_data.csv')

X_train = training_data[COLS_FEATURES]
X_test = test_data[COLS_FEATURES]

y_train = training_data[COLS_LABELS]
y_test = test_data[COLS_LABELS]



Rfr = RandomForestRegressor(n_estimators=100, max_depth=40, min_samples_split=20, min_samples_leaf=10, random_state=42)
Rfr.fit(X_train, y_train)

y_pred_test = Rfr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
y_pred_train = Rfr.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
print(f"The Test-MSE is: {mse_test:.2f}, the test-mean error is thus roughly {np.sqrt(mse_test):.2f}")
print(f"The Train-MSE is: {mse_train:.2f}, the train-mean error is thus roughly {np.sqrt(mse_train):.2f}")

# Save the model
with open('RandomForest_model_placeholder.pkl', 'wb') as file:
    pickle.dump(Rfr, file)