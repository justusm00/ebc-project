import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import pickle


from modules.util import gap_filling_mlp, gap_filling_rf, get_month_day_from_day_of_year, compute_test_loss_rf
from modules.columns import COLS_KEY, COLS_KEY_ALT
from modules.paths import PATH_PREPROCESSED, PATH_GAPFILLED, PATH_MODEL_SAVES_MLP, PATH_MODEL_SAVES_RF
from modules.MLPstuff import MLP


# SPECIFY THESE
filename_mlp = 'mlp_60_4_JM_minmax_01b3187c62d1a0ed0d00b5736092b0d1.pth'
filename_rf = 'RandomForest_model_1e9d20aaf5beee7e5792f7b4e55dc67b.pkl'
filename_mlpsw = 'mlp_60_4_JM_minmax_d2b43b2dba972e863e8a9a0deeaebbda.pth'
# filename_mlpsw = None

path_data = PATH_PREPROCESSED + 'data_merged_with_nans.csv'


def load_mlp(filename_mlp, device='cpu'):
    """Helper function to load the MLP, features, labels and trainset statistics

    Args:
        filename_mlp (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        modules.MLPstuff.MLP : MLP
        list of str : columns used as training features
        list of str : columns used as labels
        bool : whether or not normalization was used
        bool : whether or not minmax scaling was used
        list of float : means of training features (None if normalization was not used)
        list of float : std of training features (None if normalization was not used)
        list of float : mins of training features (None if minmax scaling was not used)
        list of float : maxs of training features (None if minmax scaling was not used)
        

    """
    # extract number of hidden units, hidden layers, whether normalization was used, who trained the mlp
    parts = filename_mlp.rstrip('.pth').split('_')
    num_hidden_units = int(parts[1])
    num_hidden_layers = int(parts[2])
    mlp_hash = parts[-1]
    normalization = 'norm' in parts
    minmax_scaling = 'minmax' in parts
    if (minmax_scaling is True ) and (normalization is True ) :
        raise ValueError("Can only perform normalization OR minmax_scaling")
    
    path_mlp = PATH_MODEL_SAVES_MLP + filename_mlp

    trainset_means = None
    trainset_stds = None
    trainset_mins = None
    trainset_maxs = None

    mlp_name = filename_mlp.rstrip('.pth')
    # load MLP features and labels
    with open('model_saves/features/' + mlp_hash + '.json', 'r') as file:
        cols_features = json.load(file)
    with open('model_saves/labels/' + mlp_hash + '.json', 'r') as file:
        cols_labels = json.load(file)

    # Load the MLP 
    mlp = MLP(len(cols_features), len(cols_labels), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers)
    mlp.load_state_dict(torch.load(path_mlp, map_location=torch.device(device)))

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


    return mlp, cols_features, cols_labels, normalization, minmax_scaling, trainset_means, trainset_stds, trainset_mins, trainset_maxs


def get_table_key(cols_features):
    """Get table key from list of features or return error if key is neither COLS_KEY nor COLS_KEY_ALT

    Args:
        cols_features (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # check which key was used 
    if all(elem in cols_features for elem in COLS_KEY):
        cols_key = COLS_KEY
    elif all(elem in cols_features for elem in COLS_KEY_ALT):
        cols_key = COLS_KEY_ALT
    else:
        raise ValueError("Model features do not contain a valid key")
    return cols_key
    

    
def set_default_key(df, cols_key):
    """Set default key for merging if alt key was used

    Args:
        df (_type_): _description_
        cols_key (_type_): _description_
    """
    if cols_key == COLS_KEY_ALT:
        df[['month', 'day']] = df.apply(get_month_day_from_day_of_year, axis=1)
        df = df.drop("day_of_year", axis=1)
    return df
        




def fill_gaps(path_data, filename_mlp, filename_rf, filename_mlpsw=None, diurnal_fill=None,
              suffix_mlp='_f_mlp', suffix_mlpsw='_f_mlpsw'):
    """Perform gapfilling on data using pretrained mlp. Optionally, use MLP trained only on keys and shortwave radiation to fill gaps where no other meteo data is available.

    Args:
        path_data (str): path to data (labeled and unlabeled)
        filename_mlp (str): name of file containing MLP parameters
        filename_rf (str): name of file containing RF parameters
        filename_mlpsw (str): name of file containing the MLP trained only on shortwave radiation and keys (optional)
    """
    rf_name = filename_rf.rstrip('.pkl')
    rf_hash = rf_name.split('_')[-1]
    path_rf = PATH_MODEL_SAVES_RF + filename_rf

    # load random forest model
    with open(path_rf, 'rb') as f:
        rf = pickle.load(f)


    # load RF features and labels
    with open('model_saves/features/' + rf_hash + '.json', 'r') as file:
        cols_features_rf = json.load(file)
    with open('model_saves/labels/' + rf_hash + '.json', 'r') as file:
        cols_labels_rf = json.load(file)


    # print RF test loss
    #loss_test_rf = compute_test_loss_rf(rf, filename_rf)
    #print(f"Test MSE for RF: {loss_test_rf}")



    # load MLPs
    mlp, cols_features_mlp, cols_labels_mlp, normalization, minmax_scaling, trainset_means,\
        trainset_stds, trainset_mins, trainset_maxs  = load_mlp(filename_mlp)
    #loss_test_mlp = compute_test_loss_mlp(mlp, filename_mlp)
    #print(f"Test MSE for MLP trained on {cols_features_mlp}: {loss_test_mlp:.2f}")

    if filename_mlpsw:
        mlpsw, cols_features_mlpsw, cols_labels_mlpsw, normalization_sw, minmax_scaling_sw, \
            trainset_means_sw, trainset_stds_sw, trainset_mins_sw, trainset_maxs_sw  = load_mlp(filename_mlpsw)
        #loss_test_mlpsw = compute_test_loss_mlp(mlpsw, filename_mlpsw)
        #print(f"Test MSE for MLP trained on {cols_features_mlpsw}: {loss_test_mlpsw:.2f}")



    if (set(cols_labels_rf) != set(cols_labels_mlp)):
        raise ValueError("Labels / target variables must be the same for all models")
    if filename_mlpsw:
        if (set(cols_labels_mlpsw) != set(cols_labels_mlp)):
            raise ValueError("Labels / target variables must be the same for all models")
        
    # now only use single variable for labels
    cols_labels = cols_labels_rf

    

    # get table keys for merging
    cols_key_rf = get_table_key(cols_features_rf)
    cols_key_mlp = get_table_key(cols_features_mlp)
    if filename_mlpsw:
        cols_key_mlpsw = get_table_key(cols_features_mlpsw)
      
    # load data
    data = pd.read_csv(path_data)


    # get both gapfilled dataframes
    df_mlp = gap_filling_mlp(data=data, mlp=mlp, columns_key=cols_key_mlp, cols_features=cols_features_mlp,
                             cols_labels=cols_labels, suffix = suffix_mlp, means=trainset_means, stds=trainset_stds,
                             mins=trainset_mins, maxs=trainset_maxs)

    df_rf = gap_filling_rf(data=data, model=rf, columns_key=cols_key_rf, cols_features=cols_features_rf, cols_labels=cols_labels)


    if filename_mlpsw:
        df_mlpsw = gap_filling_mlp(data=data, mlp=mlpsw, columns_key=cols_key_mlpsw, cols_features=cols_features_mlpsw,
                                   cols_labels=cols_labels, suffix=suffix_mlpsw, means=trainset_means_sw, stds=trainset_stds_sw,
                                   mins=trainset_mins_sw, maxs=trainset_maxs_sw)
    


    # make sure all have the same key so that they can be merged (just use the default key)
    df_mlp = set_default_key(df_mlp, cols_key_mlp)
    df_rf = set_default_key(df_rf, cols_key_rf)
    if filename_mlpsw:
        df_mlpsw = set_default_key(df_mlpsw, cols_key_mlpsw)

    cols_key_mlpsw = COLS_KEY
    cols_key_mlp = COLS_KEY
    cols_key_rf = COLS_KEY

    cols_gapfilled_mlp = [col.replace('_orig', '') + suffix_mlp for col in cols_labels]
    cols_gapfilled_mlpsw = [col.replace('_orig', '') + suffix_mlpsw for col in cols_labels]
    cols_gapfilled_mds = [col.replace('_orig', '') + '_f' for col in cols_labels]

    # merge mlp and and rf
    df = df_mlp[COLS_KEY + cols_gapfilled_mlp].merge(df_rf, how="outer", on=COLS_KEY)

    # print NaN statistics
    print("Total number of records: \t \t", df.shape[0])
    for col_mlp, col_mds in zip(cols_gapfilled_mlp, cols_gapfilled_mds):
        print(f"Number of NaNs in {col_mlp}: \t \t {df[df[col_mlp].isna()].shape[0]}")
        print(f"Number of NaNs in {col_mds}: \t \t \t {df[df[col_mds].isna()].shape[0]}")

    # merge mlp sw on top 
    if filename_mlpsw:
        df = df_mlpsw[COLS_KEY + cols_gapfilled_mlpsw].merge(df, how="outer", on=COLS_KEY)
        # Now fill the dataframe
        for col_mlp, col_mlpsw in zip(cols_gapfilled_mlp, cols_gapfilled_mlpsw):
            df[col_mlp] = df[col_mlp]\
                .fillna( df[col_mlpsw])
            df = df.drop(col_mlpsw, axis=1)
            print(f"Number of NaNs in {col_mlp} after adding data from MLP trained on {cols_features_mlpsw}: \t \t {df[df[col_mlp].isna()].shape[0]}")



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
    fill_gaps(path_data=path_data, filename_mlp=filename_mlp, filename_rf=filename_rf, filename_mlpsw=filename_mlpsw)