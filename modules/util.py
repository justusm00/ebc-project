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

