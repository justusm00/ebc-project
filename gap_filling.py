import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import pickle


from modules.util import gap_filling_mlp, gap_filling_rf, get_month_day_from_day_of_year, compute_test_loss_mlp, compute_test_loss_rf
from columns import COLS_KEY, COLS_KEY_ALT
from paths import PATH_PREPROCESSED, PATH_GAPFILLED, PATH_MODEL_SAVES_MLP, PATH_MODEL_SAVES_RF
from modules.MLPstuff import MLP


# SPECIFY THESE
filename_mlp = 'mlp_60_4_JM_minmax_01b3187c62d1a0ed0d00b5736092b0d1.pth'
filename_rf = 'RandomForest_model_ae6a618e4da83a56de13c7eec7152215.pkl'

path_data = PATH_PREPROCESSED + 'data_merged_with_nans.csv'


def fill_gaps(path_data, filename_mlp, filename_rf):
    """Perform gapfilling on data using pretrained mlp. 

    Args:
        path_data (str): path to data (labeled and unlabeled)
        filename_mlp (str): name of file containing MLP parameters
        filename_rf (str): name of file containing RF parameters
    """

    # extract number of hidden units, hidden layers, whether normalization was used, who trained the mlp
    parts = filename_mlp.split('_')
    num_hidden_units = int(parts[1])
    num_hidden_layers = int(parts[2])
    normalization = 'norm' in parts
    minmax_scaling = 'minmax' in parts
    if (minmax_scaling is True ) and (normalization is True ) :
        raise ValueError("Can only perform normalization OR minmax_scaling")
    path_mlp = PATH_MODEL_SAVES_MLP + filename_mlp
    path_rf = PATH_MODEL_SAVES_RF + filename_rf
    trainset_means = None
    trainset_stds = None
    trainset_mins = None
    trainset_maxs = None


    mlp_name = filename_mlp.rstrip('.pth')
    rf_name = filename_rf.rstrip('.pkl')

    # load MLP features and labels
    with open(PATH_MODEL_SAVES_MLP + 'features/' + mlp_name + '.json', 'r') as file:
        cols_features_mlp = json.load(file)
    with open(PATH_MODEL_SAVES_MLP + 'labels/' + mlp_name + '.json', 'r') as file:
        cols_labels_mlp = json.load(file)


    # load RF features and labels
    with open(PATH_MODEL_SAVES_RF + 'features/' + rf_name + '.json', 'r') as file:
        cols_features_rf = json.load(file)
    with open(PATH_MODEL_SAVES_RF + 'labels/' + rf_name + '.json', 'r') as file:
        cols_labels_rf = json.load(file)




    if set(cols_labels_rf) != set(cols_labels_mlp):
        raise ValueError("RF and MLP labels are not the same.")
    


    # check which key was used for MLP
    if all(elem in cols_features_mlp for elem in COLS_KEY):
        cols_key_mlp = COLS_KEY
    elif all(elem in cols_features_mlp for elem in COLS_KEY_ALT):
        cols_key_mlp = COLS_KEY_ALT
    else:
        raise ValueError("MLP features do not contain a valid key")
    

    # check which key was used for RF
    if all(elem in cols_features_rf for elem in COLS_KEY):
        cols_key_rf = COLS_KEY
    elif all(elem in cols_features_rf for elem in COLS_KEY_ALT):
        cols_key_rf = COLS_KEY_ALT
    else:
        raise ValueError("RF features do not contain a valid key")
    

    # Load the MLP mlp
    mlp = MLP(len(cols_features_mlp), len(cols_labels_mlp), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers)
    mlp.load_state_dict(torch.load(path_mlp))


    # load statistics
    if normalization:
        # load statistics
        model_means_path = PATH_MODEL_SAVES_MLP + 'statistics/' + mlp_name + '_means.npy'
        model_stds_path = PATH_MODEL_SAVES_MLP + 'statistics/' + mlp_name + '_stds.npy'
        trainset_means = np.load(model_means_path)
        trainset_stds = np.load(model_stds_path)


    if minmax_scaling:
        model_maxs_path = PATH_MODEL_SAVES_MLP + 'statistics/' + mlp_name + '_maxs.npy'
        model_mins_path = PATH_MODEL_SAVES_MLP + 'statistics/' + mlp_name + '_mins.npy'
        trainset_maxs = np.load(model_maxs_path)
        trainset_mins = np.load(model_mins_path)


    # load random forest model
    with open(path_rf, 'rb') as f:
        rf = pickle.load(f)


    # print losses of models
    loss_test_mlp = compute_test_loss_mlp(mlp, cols_features_mlp, cols_labels_mlp, normalization, minmax_scaling)
    loss_test_rf = compute_test_loss_rf(rf, cols_features_rf, cols_labels_rf)

    print(f"Test MSE for RF: {loss_test_rf:.2f}, Test MSE for MLP: {loss_test_mlp:.2f}")



    # load data
    data = pd.read_csv(path_data)


    # get both gapfilled dataframes
    df_mlp = gap_filling_mlp(data=data, mlp=mlp, columns_key=cols_key_mlp, columns_data=cols_features_mlp,
                             columns_labels=cols_labels_mlp, means=trainset_means, stds=trainset_stds,
                             mins=trainset_mins, maxs=trainset_maxs)

    df_rf = gap_filling_rf(data=data, model=rf, columns_key=cols_key_rf, columns_data=cols_features_rf, columns_labels=cols_labels_rf)


    # make sure both have the same key so that they can be merged (just use the default key)
    if cols_key_mlp == COLS_KEY_ALT:
        df_mlp[['month', 'day']] = df_mlp.apply(get_month_day_from_day_of_year, axis=1)
        df_mlp = df_mlp.drop("day_of_year", axis=1)
    if cols_key_rf == COLS_KEY_ALT:
        df_rf[['month', 'day']] = df_rf.apply(get_month_day_from_day_of_year, axis=1)
        df_rf = df_rf.drop("day_of_year", axis=1)



    # merge both
    df = df_mlp[COLS_KEY + [col.replace('_orig', '') + '_f_mlp' for col in cols_labels_mlp]].merge(df_rf, how="inner", on=COLS_KEY)

    # Convert the '30min' column to a timedelta representing the minutes
    df['time'] = pd.to_timedelta(df['30min'] * 30, unit='m')

    # Create the datetime column by combining 'year', 'month', 'day' and 'time'
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']]) + df['time']

    # drop year, month, day columns
    df = df.drop(['year', 'month', 'day', '30min', 'time'], axis=1)


    # filter by location and sort by timestamps
    df_bg = df[df['location'] == 0].sort_values(by='timestamp')
    df_gw = df[df['location'] == 1].sort_values(by='timestamp')

    # drop location columns
    df_bg = df_bg.drop('location', axis=1)
    df_gw = df_gw.drop('location', axis=1)


    # save files
    df_bg.to_csv(PATH_GAPFILLED + 'BG_gapfilled.csv', index=False)
    df_gw.to_csv(PATH_GAPFILLED + 'GW_gapfilled.csv', index=False)



if __name__ == '__main__':
    fill_gaps(path_data=path_data, filename_mlp=filename_mlp, filename_rf=filename_rf)