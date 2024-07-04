from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib as plt
import pickle

from modules.util import grab_data

from columns import COLS_FEATURES, COLS_DAYOFYEAR

input_data, target_data, dim_in, dim_out = grab_data(path='data/training_data_merged.csv', columns_data=COLS_FEATURES+COLS_DAYOFYEAR, columns_labels=['H_orig', 'LE_orig'], return_dataset=False)

input_data.head()
target_data.head()

# First look into efficiency of random forest
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

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