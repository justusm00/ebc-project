import json
import torch
import numpy as np
import pickle
import pandas as pd

from modules.paths import PATH_MODEL_SAVES_MLP,\
    PATH_MODEL_SAVES_RF, PATH_MODEL_SAVES_FEATURES, PATH_MODEL_SAVES_LABELS, PATH_MODEL_SAVES_STATISTICS    
from modules.util import extract_mlp_details_from_name
from modules.MLPstuff import MLP


def replace_filled_with_original(row, col, col_f):
    """Where original data is available, replace gapfilled data with it

    Args:
        row (_type_): _description_
        col (_type_): _description_
        col_f (_type_): _description_

    Returns:
        _type_: _description_
    """
    if pd.isna(row[col]):
        return row[col_f]
    else:
        return row[col]


def gap_filling_mlp(data, mlp, cols_features, cols_labels, suffix='_f_mlp', means=None, stds=None,
                     mins=None, maxs=None):
    """Fill gaps using pretrained model.

    Args:
        data (pd.DataFrame): Pandas dataframe containing prediction data
        mlp (modules.MLPstuff.MLP): model
        cols_features (list): list of column names used for training
        cols_labels (list): list of column names to predict
        suffix (str): suffix of gapfilled columns (e.g. H_orig is changed to H_f_mlp per default)
        means (list, optional): List of means, must be provided if training was done on normalized data. Defaults to None.
        stds (list, optional): List of standard deviations, must be provided if training was done on normalized data. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing the original data and the MLP gap filled data.
    """

    if ((means is None) and (stds is not None)) or ((stds is None) and (means is not None)) :
        raise ValueError("Must specify either means and stds or none of them.")
    if ((mins is None) and (maxs is not None)) or ((maxs is None) and (mins is not None)) :
        raise ValueError("Must specify either mins and maxs or none of them.")
    if ((means is not None) and (maxs is not None)) or ((means is not None) and (mins is not None)) :
        raise ValueError("If means and stds are specified, mins and maxs must be None and vice versa")
    
    # data used for prediction
    input = data[cols_features]

    # transform input into torch.tensor and make predictions
    input_tensor = torch.tensor(input.values, dtype=torch.float32)

    # normalize
    if means is not None and stds is not None:
        means = torch.tensor(means)
        stds = torch.tensor(stds)
        input_tensor = (input_tensor - means) / stds

    # minmax scaling
    if mins is not None and maxs is not None:
        mins = torch.tensor(mins)
        maxs = torch.tensor(maxs)
        input_tensor = (input_tensor - mins) / (maxs - mins)

    with torch.no_grad():
        pred = mlp(input_tensor).numpy() #  Transform back to numpy 

    data_pred = merge_predictions_on_data(data, pred, cols_features, cols_labels, suffix)

    return data_pred



def gap_filling_rf(data, model, cols_features, cols_labels, suffix='_f_rf'):
    """Fill gaps using pretrained Random Forest model.

    Args:
        data (pd.DataFrame): Pandas dataframe containing prediction data
        model (_type_): RandomForestRegressor
        cols_features (list): list of column names used for training
        cols_labels (list): list of column names to predict
        suffix (str): suffix added to the RF gapfilled columns

    Returns:
        pd.DataFrame: Dataframe containing the original data and the MLP gap filled data.
    """

    # data used for prediction
    X = data[cols_features]

    # create dataframe of predictions 
    y = model.predict(X)
    

    data_pred = merge_predictions_on_data(data, y, cols_features, cols_labels, suffix)


    return data_pred


def merge_predictions_on_data(data, pred, cols_features, cols_labels, suffix):
    """Merge MLP / RF predictions on data and replace gapfilled data with original data where available.

    Args:
        data (_type_): _description_
        pred (_type_): _description_
        cols_labels (_type_): _description_
        suffix (_type_): _description_

    Returns:
        _type_: _description_
    """
    cols_labels_pred = [col.replace('_orig', '') + suffix  for col in cols_labels]
    pred = pd.DataFrame(pred, columns=cols_labels_pred)


    # merge predictions onto features
    data_pred = pd.concat([data, pred], axis=1)

    # make sure that predictions are na if one or more features are na
    data_pred.loc[data_pred[cols_features].isna().any(axis=1), cols_labels_pred] = pd.NA

    # replace values with original values where available
    for col_f, col in zip(cols_labels_pred, cols_labels):
        data_pred[col_f] = data_pred.apply(replace_filled_with_original, axis=1, args=(col, col_f))

    return data_pred

def load_mlp(filename, device='cpu'):
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
    name = filename.rstrip('.pth')

    num_hidden_units, num_hidden_layers, model_hash,\
        cols_features, cols_labels, normalization, minmax_scaling = extract_mlp_details_from_name(name)
    
    path = PATH_MODEL_SAVES_MLP + filename

    trainset_means = None
    trainset_stds = None
    trainset_mins = None
    trainset_maxs = None

    # load MLP features and labels
    with open(PATH_MODEL_SAVES_FEATURES + model_hash + '.json', 'r') as file:
        cols_features = json.load(file)
    with open(PATH_MODEL_SAVES_LABELS + model_hash + '.json', 'r') as file:
        cols_labels = json.load(file)

    # Load the MLP 
    mlp = MLP(len(cols_features), len(cols_labels), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers)
    mlp.load_state_dict(torch.load(path, map_location=torch.device(device)))

    # load statistics
    if normalization:
        # load statistics
        model_means_path = PATH_MODEL_SAVES_STATISTICS + model_hash + '_means.npy'
        model_stds_path = PATH_MODEL_SAVES_STATISTICS + model_hash  + '_stds.npy'
        trainset_means = np.load(model_means_path)
        trainset_stds = np.load(model_stds_path)


    if minmax_scaling:
        model_maxs_path = PATH_MODEL_SAVES_STATISTICS + model_hash + '_maxs.npy'
        model_mins_path = PATH_MODEL_SAVES_STATISTICS + model_hash + '_mins.npy'
        trainset_maxs = np.load(model_maxs_path)
        trainset_mins = np.load(model_mins_path)


    return mlp, cols_features, cols_labels, model_hash, normalization, minmax_scaling, trainset_means, trainset_stds, trainset_mins, trainset_maxs


def load_rf(filename):
    """Load random forest model, hash, features and labels

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    name = filename.rstrip('.pkl')
    model_hash = name.split('_')[-1]
    path = PATH_MODEL_SAVES_RF + filename

    # load random forest model
    with open(path, 'rb') as f:
        rf = pickle.load(f)

    # load RF features and labels
    with open(PATH_MODEL_SAVES_FEATURES + model_hash + '.json', 'r') as file:
        cols_features = json.load(file)
    with open(PATH_MODEL_SAVES_LABELS + model_hash + '.json', 'r') as file:
        cols_labels = json.load(file)
    return rf, model_hash, cols_features, cols_labels