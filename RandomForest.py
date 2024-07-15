from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle
import json


from modules.columns import COLS_FEATURES_ALL, COLS_LABELS_ALL
from modules.paths import PATH_MODEL_TRAINING, PATH_MODEL_SAVES_RF, PATH_PREPROCESSED
from modules.util import get_hash_from_features_and_labels
from modules.dataset_util import train_test_splitter


# ALWAYS SPECIFY THESE
# cols_features = ["incomingShortwaveRadiation", "location", "soilTemperature",
#                  "windSpeed", "airTemperature", "30min", "day_of_year"]
cols_features = COLS_FEATURES_ALL
#cols_features = ["30min"]
cols_labels = COLS_LABELS_ALL




def fit_rf(cols_features, cols_labels, save_results=True, verbose=True):
    """Fit random forest model, print train/test MSEs and save model.

    Args:
        cols_features (_type_): _description_
        cols_labels (_type_): _description_
        path_model_saves (_type_): _description_
        save_train_test_data (bool): if set to True, the model and the train / test data are saved

    Raises:
        ValueError: _description_
    """
    # sort features and labels
    cols_features = sorted(cols_features)
    cols_labels = sorted(cols_labels)

    path_save = None
    if save_results:
        path_save = PATH_MODEL_TRAINING
    # Create a hash based on the features and labels
    model_hash = get_hash_from_features_and_labels(cols_features=cols_features, cols_labels=cols_labels)

    # create model name
    model_name = f'RandomForest_model_{model_hash}'

    if verbose:
        print("\n")
        print(f"Features used: {len(cols_features)} ({cols_features}) \n")
        print(f"Labels used: {len(cols_labels)} ({cols_labels}) \n")

    if save_results:
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
        if verbose:
            print("No train and test data available for given feature/label combination. Creating one ... \n")
        training_data, test_data = train_test_splitter(path_data=PATH_PREPROCESSED + 'data_merged_with_nans.csv', 
                               cols_features=cols_features, 
                               cols_labels=cols_labels, 
                               model_hash=model_hash,
                               path_save=path_save,
                               verbose=verbose)


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
    if verbose:
        print(f"The Test-MSE is: {mse_test:.2f}, the test-mean error is thus roughly {np.sqrt(mse_test):.2f}")
        print(f"The Train-MSE is: {mse_train:.2f}, the train-mean error is thus roughly {np.sqrt(mse_train):.2f}")

    if save_results:
        # Save the model
        with open(PATH_MODEL_SAVES_RF + model_name + '.pkl', 'wb') as file:
            pickle.dump(Rfr, file)
        
        if verbose:
            print(f"Model saved under {PATH_MODEL_SAVES_RF + model_name + '.pkl'}")

    return Rfr, model_name, model_hash, mse_test




def forward_selection_rf(max_features=None):
    selected_features = []
    best_mse = float('inf')
    best_model = None
    best_model_name = None
    best_model_hash = None
    remaining_features = COLS_FEATURES_ALL
    n_features = len(remaining_features)

    if max_features is None:
        max_features = n_features

    while remaining_features and len(selected_features) < max_features:
        mse_list = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            model, model_name, model_hash, mse = fit_rf(features_to_test, cols_labels, save_results=False, verbose=False)
            mse_list.append((mse, feature))

        mse_list.sort()
        best_new_mse, best_new_feature = mse_list[0]

        if best_new_mse < best_mse:
            best_mse = best_new_mse
            best_model = model
            best_model_name = model_name
            best_model_hash = model_hash
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            print(f"Selected feature: {best_new_feature}, MSE: {best_mse}")
        else:
            break

    print(f"MSE for best model (trained on {selected_features}) : {best_mse} \n")
    # rename best model
    best_model_name = 'best' + best_model_name
    # save best features
    features_json = 'model_saves/features/' + best_model_hash + '.json'
    labels_json = 'model_saves/labels/' + best_model_hash + '.json'
    with open(features_json, 'w') as file:
        json.dump(cols_features, file)
    with open(labels_json, 'w') as file:
        json.dump(cols_labels, file)


    # Save best model
    with open(PATH_MODEL_SAVES_RF + best_model_name + '.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    return 




def stepwise_forward_selection_rf(max_features=None, tol=0.01):
    selected_features = []
    best_mse = float('inf')
    best_model = None
    best_model_name = None
    best_model_hash = None
    remaining_features = COLS_FEATURES_ALL
    n_features = len(remaining_features)

    if max_features is None:
        max_features = n_features

    while remaining_features and len(selected_features) < max_features:
        mse_list = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            model, model_name, model_hash, mse = fit_rf(features_to_test, cols_labels, save_results=False, verbose=False)
            mse_list.append((mse, feature))

        mse_list.sort()
        best_new_mse, best_new_feature = mse_list[0]

        if best_new_mse < best_mse:
            best_mse = best_new_mse
            best_model = model
            best_model_name = model_name
            best_model_hash = model_hash
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            print(f"Selected feature: {best_new_feature}, MSE: {best_mse}")

            # Stepwise part: check if any of the already selected features can be removed
            if len(selected_features) > 1:
                improved = True
                while improved:
                    improved = False
                    for feature in selected_features:
                        features_to_test = [f for f in selected_features if f != feature]
                        model, model_name, model_hash, mse = fit_rf(features_to_test, cols_labels, save_results=False, verbose=False)
                        # check if relative change in mse is smaller than tol
                        if abs(mse - best_mse) / best_mse < tol:
                            best_mse = mse
                            best_model = model
                            best_model_name = model_name
                            best_model_hash = model_hash
                            selected_features.remove(feature)
                            improved = True
                            print(f"Removed feature: {feature}, MSE: {best_mse}")
                            break
        else:
            break

    print(f"MSE for best model (trained on {selected_features}) : {best_mse} \n")
    
    # Rename and save the best model
    best_model_name = 'best_' + best_model_name
    with open(PATH_MODEL_SAVES_RF + best_model_name + '.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    # Save best features
    features_json = 'model_saves/features/' + best_model_hash + '.json'
    labels_json = 'model_saves/labels/' + best_model_hash + '.json'
    with open(features_json, 'w') as file:
        json.dump(selected_features, file)
    with open(labels_json, 'w') as file:
        json.dump(cols_labels, file)

    return




if __name__ == '__main__':
    fit_rf(cols_features=cols_features, cols_labels=cols_labels, save_results=True, verbose=True)
    # stepwise_forward_selection_rf()
