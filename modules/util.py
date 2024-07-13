import numpy as np
import pandas as pd
import torch


import datetime
import hashlib
import json



def encode_timestamp(timestamp):
    """
    For encoding the 30 minute blocks into integers 
    """
    number_of_seconds = timestamp.hour * 3600 + timestamp.minute * 60
    return number_of_seconds // 1800 # 1800 because of 30-minute stepwidth


def transform_timestamp(df, col_name):
    """
    Transform timestamp to proper date/year/month/day values
    
    Args:
        df
        col_name: column of original dataframe based on which to infer dates
    """

    df['date'] = df[f'{col_name}'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    df['year'] = df[f'{col_name}'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S').year)
    df['month'] = df[f'{col_name}'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S').month)
    df['day'] = df[f'{col_name}'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S').day)
    df['day_of_year'] = df['date'].dt.dayofyear

    """
    Robin: Encode the 30 minute intervals using integers
    """
    df['30min'] = df['date'].apply( encode_timestamp )
    
    return df


def numerical_to_float(df, cols):
    """
    Args:
        df (pandas dataframe): dataframe to preprocess
        cols (list of str): names of columns to apply the dtype change to
    Returns:
        df (pandas dataframe)
    """
    for c in cols:
        try:
            df[f'{c}'] = df[f'{c}'].astype(dtype=float)
        except ValueError:
            # some files use ',' (comma) as decimal separator, replace with '.' (dot)
            df[f'{c}'] = df[f'{c}'].apply(lambda x: str(x).replace(',', '.'))
            df[f'{c}'] = df[f'{c}'].astype(dtype=float)
    
    return df

    


def extract_mlp_details_from_name(model_name):
    """_summary_

    Args:
        model_name (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    parts = model_name.split('_')
    num_hidden_units = int(parts[1])
    num_hidden_layers = int(parts[2])
    model_hash = parts[-1]
    normalization = 'norm' in parts
    minmax_scaling = 'minmax' in parts
    if (minmax_scaling is True ) and (normalization is True ) :
        raise ValueError("Can only perform normalization OR minmax_scaling")
    # load RF features and labels
    with open('model_saves/features/' + model_hash + '.json', 'r') as file:
        cols_features = json.load(file)
    with open('model_saves/labels/' + model_hash + '.json', 'r') as file:
        cols_labels = json.load(file)


    
    return num_hidden_units, num_hidden_layers, model_hash, cols_features, cols_labels, normalization, minmax_scaling





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

    # replace gapfilled values with original values where available
    for col_f, col in zip(cols_labels_pred, cols_labels):
        data_pred[col_f] = data_pred.apply(replace_filled_with_original, axis=1, args=(col, col_f))


    # make sure that predictions are na if one or more features are na
    data_pred.loc[data_pred[cols_features].isna().any(axis=1), cols_labels_pred] = pd.NA

    return data_pred


def get_hash_from_features_and_labels(cols_features, cols_labels):
    """Creates a unique hash for each combination of features and labels, used to save models. 

    Args:
        cols_features (_type_): _description_
        cols_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Convert the list to a sorted tuple to ensure order doesn't matter
    list_conc = cols_features + cols_labels
    sorted_list = tuple(sorted(list_conc))
    
    # Create a hash from the sorted tuple
    list_str = str(sorted_list)  # Convert the sorted tuple to a string
    return hashlib.md5(list_str.encode()).hexdigest()  # Use MD5 hash


# Function to calculate day of the year
def get_day_of_year(row):
    date = datetime.datetime(row['year'], row['month'], row['day'])
    return date.timetuple().tm_yday


def get_month_day_from_day_of_year(row):
    year = int(row['year'])
    day_of_year = int(row['day_of_year'])
    date_obj = datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    return pd.Series([date_obj.month, date_obj.day])

