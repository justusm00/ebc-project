from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle
import json


from modules.columns import COLS_LABELS_ALL, COLS_KEY, COLS_KEY_ALT
from modules.paths import PATH_MODEL_TRAINING, PATH_MODEL_SAVES_RF, PATH_PREPROCESSED
from modules.util import get_hash_from_features_and_labels
from modules.dataset import train_test_splitter


# ALWAYS SPECIFY THESE
cols_features = COLS_KEY_ALT + ["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit"]
# cols_features = COLS_KEY_ALT + ["incomingShortwaveRadiation"]
cols_labels = COLS_LABELS_ALL




def fit_rf(cols_features, cols_labels):
    """Fit random forest model, print train/test MSEs and save model.

    Args:
        cols_features (_type_): _description_
        cols_labels (_type_): _description_
        path_model_saves (_type_): _description_

    Raises:
        ValueError: _description_
    """
    # Create a hash based on the features and labels
    model_hash = get_hash_from_features_and_labels(cols_features=cols_features, cols_labels=cols_labels)

    # create model name
    model_name = f'RandomForest_model_{model_hash}'

    print("\n")
    print(f"Features used: {len(cols_features)} ({cols_features}) \n")
    print(f"Labels used: {len(cols_labels)} ({cols_labels}) \n")

    # save features and labels
    features_json = 'model_saves/features/' + model_hash + '.json'
    labels_json = 'model_saves/labels/' + model_hash + '.json'
    with open(features_json, 'w') as file:
        json.dump(cols_features, file)
    with open(labels_json, 'w') as file:
        json.dump(cols_labels, file)

    try:
        training_data = pd.read_csv(PATH_MODEL_TRAINING + 'training_data_' + model_hash + '.csv')
        test_data = pd.read_csv(PATH_MODEL_TRAINING + 'test_data_' + model_hash + '.csv')
    except:
        print("No train and test data available for given feature/label combination. Creating one ... \n")
        train_test_splitter(path_data=PATH_PREPROCESSED + 'data_merged_with_nans.csv', 
                               cols_features=cols_features, 
                               cols_labels=cols_labels, 
                               path_save=PATH_MODEL_TRAINING, model_hash=model_hash)
        training_data = pd.read_csv(PATH_MODEL_TRAINING + 'training_data_' + model_hash + '.csv')
        test_data = pd.read_csv(PATH_MODEL_TRAINING + 'test_data_' + model_hash + '.csv')


    X_train = training_data[cols_features]
    X_test = test_data[cols_features]

    y_train = training_data[cols_labels]
    y_test = test_data[cols_labels]



    Rfr = RandomForestRegressor(n_estimators=100, max_depth=40, min_samples_split=20,
                                 min_samples_leaf=10, random_state=42)
    Rfr.fit(X_train, y_train)

    y_pred_test = Rfr.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    y_pred_train = Rfr.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    print(f"The Test-MSE is: {mse_test:.2f}, the test-mean error is thus roughly {np.sqrt(mse_test):.2f}")
    print(f"The Train-MSE is: {mse_train:.2f}, the train-mean error is thus roughly {np.sqrt(mse_train):.2f}")

    # Save the model
    with open(PATH_MODEL_SAVES_RF + model_name + '.pkl', 'wb') as file:
        pickle.dump(Rfr, file)

    print(f"Model saved under {PATH_MODEL_SAVES_RF + model_name + '.pkl'}")


if __name__ == '__main__':
    fit_rf(cols_features=cols_features, cols_labels=cols_labels)
