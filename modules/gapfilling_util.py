import json
import torch
import numpy as np
import pickle

from modules.paths import PATH_MODEL_SAVES_MLP,\
    PATH_MODEL_SAVES_RF, PATH_MODEL_SAVES_FEATURES, PATH_MODEL_SAVES_LABELS
from modules.util import extract_mlp_details_from_name
from modules.MLPstuff import MLP

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
        model_means_path = PATH_MODEL_SAVES_MLP + 'statistics/' + name + '_means.npy'
        model_stds_path = PATH_MODEL_SAVES_MLP + 'statistics/' + name + '_stds.npy'
        trainset_means = np.load(model_means_path)
        trainset_stds = np.load(model_stds_path)


    if minmax_scaling:
        model_maxs_path = PATH_MODEL_SAVES_MLP + 'statistics/' + name + '_maxs.npy'
        model_mins_path = PATH_MODEL_SAVES_MLP + 'statistics/' + name + '_mins.npy'
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