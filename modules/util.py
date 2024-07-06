import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from modules.MLPstuff import MLP
import datetime
import hashlib



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


############################# MLP STUFF ####################################

# Define Dataset
class EBCDataset(Dataset):
    def __init__(self, data, labels, normalization=False, means=None, stds=None):
        if (normalization is True ) and ( means is None or stds is None) :
            raise ValueError("Must specify mean and standard for normalization.")
        self.data = data
        self.labels = labels
        self.normalization = normalization
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.normalization:
            sample = (sample - self.means) / self.stds 
        return sample, label



# Data loader
def grab_data(path_train, path_test, num_cpus, columns_data=None, columns_labels=None, normalization=False):
    """Loads training and test data from respective directories. 

    Args:
        path_train (_type_): _description_
        path_test (_type_): _description_
        num_cpus (_type_): _description_
        columns_data (_type_, optional): _description_. Defaults to None.
        columns_labels (_type_, optional): _description_. Defaults to None.
        normalization (bool, optional): If True, normalize data based on trainset statistics.

    Returns:
        _type_: _description_
    """

    # load data
    trainset = pd.read_csv(path_train)
    testset = pd.read_csv(path_test)

    # Select data and labels
    if columns_data == None:
        columns_data = ['CO2', 'H2O', 'Ustar', 'location', 'year', 'month', 'day', '30min']
    if columns_labels == None:
        columns_labels = ['H_orig', 'LE_orig']
   

    # Convert to torch tensor
    trainset_data = torch.tensor(trainset[ columns_data ].values, dtype=torch.float32)
    trainset_labels = torch.tensor(trainset[ columns_labels].values, dtype=torch.float32)
    testset_data = torch.tensor(testset[ columns_data ].values, dtype=torch.float32)
    testset_labels = torch.tensor(testset[ columns_labels].values, dtype=torch.float32)
    # first load unnormalized datasets
    trainset = EBCDataset(trainset_data, trainset_labels, normalization=False)
    testset = EBCDataset(testset_data, testset_labels, normalization=False)
    if normalization: # normalize to [-1,1]
        num_samples = trainset.data.shape[0]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_samples, 
                                                    num_workers=num_cpus)
        
        # compute column means and stds
        rows, _ = next(iter(trainloader))
        trainset_mean = torch.mean(rows, dim=(0)) 
        trainset_std = torch.std(rows, dim=(0))

        # load again, now normalized
        trainset = EBCDataset(trainset_data, trainset_labels, normalization=True, means=trainset_mean, stds=trainset_std)
        testset = EBCDataset(testset_data, testset_labels, normalization=True, means=trainset_mean, stds=trainset_std)
        

    return trainset, testset
    

    



# dataset Splitter 
def train_val_splitter(dataset, split_seed=42, val_frac = 0.2):
    """Splits the given dataset into training and validation sets.

    Args:
        dataset (_type_): _description_
        split_seed (int, optional): _description_. Defaults to 42.
        val_frac (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    
    # Train Val Split
    num_val_samples = np.ceil(val_frac * len(dataset)).astype(int)
    num_train_samples = len(dataset) - num_val_samples
    trainset, valset = torch.utils.data.random_split(dataset, 
                                                    (num_train_samples, num_val_samples), 
                                                    generator=torch.Generator().manual_seed(split_seed))
    
    return trainset, valset


# Dataloaders
def data_loaders(trainset, valset, testset, batch_size=64, num_cpus=1):
    """Initialize train, validation and test data loader.

    Args:
        trainset: Training set torchvision dataset object.
        valset: Validation set torchvision dataset object.
        testset: Test set torchvision dataset object.
        batch_size: Batchsize used during training, defaults to 64
        num_cpus: Number of CPUs to use when iterating over
            the data loader. More is faster. Defaults to 1.
    """        
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_cpus, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_cpus, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True, 
                                             num_workers=num_cpus, pin_memory=True)
    return trainloader, valloader, testloader



def gap_filling_mlp(data, mlp, columns_key, columns_data, columns_labels, means=None, stds=None):
    """Fill gaps using pretrained model.

    Args:
        data (pd.DataFrame): Pandas dataframe containing prediction data
        mlp (modules.MLPstuff.MLP): model
        columns_key (list): columns that uniquely identify a record
        columns_data (list): list of column names used for training
        columns_labels (list): list of column names to predict
        means (list, optional): List of means, must be provided if training was done on normalized data. Defaults to None.
        stds (list, optional): List of standard deviations, must be provided if training was done on normalized data. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing the original data and the MLP gap filled data.
    """

    if ((means is None) and (stds is not None)) or ((stds is None) and (means is not None)) :
        raise ValueError("Must specify either means and stds or none of them.")
    # identify rows where labels are NaN, but features aren't
    mask_nan = data[columns_labels].isna().any(axis=1)
    mask_not_nan = data[columns_data].notna().all(axis=1)

    # Combine the masks
    combined_mask = mask_nan & mask_not_nan

    # data used for prediction
    input = data[combined_mask][columns_data].reset_index(drop=True)

    # transform input into torch.tensor and make predictions
    input_tensor = torch.tensor(input.values, dtype=torch.float32)

    # normalize
    if means is not None and stds is not None:
        means = torch.tensor(means)
        stds = torch.tensor(stds)
        input_tensor = (input_tensor - means) / stds

    with torch.no_grad():
        pred = mlp(input_tensor).numpy() #  Transform back to numpy 

    # create dataframe of predictions 
    columns_labels_pred = [col.replace('_orig', '') + '_f_mlp'  for col in columns_labels]
    pred = pd.DataFrame(pred, columns=columns_labels_pred)

    
    # merge predictions onto features
    data_pred = pd.concat([input, pred], axis=1)


    # merge both dataframes
    data_merged = data.merge(data_pred[columns_key + columns_labels_pred], how="outer", on=columns_key)

    # now, the gapfilled columns have nan values where the original data is not nan. In this case, just take the original values
    for col_f, col in zip(columns_labels_pred, columns_labels):
        data_merged[col_f] = data_merged[col_f].fillna(data_merged[col])


    return data_merged



def gap_filling_rf(data, model, columns_key, columns_data, columns_labels):
    """Fill gaps using pretrained Random Forest model.

    Args:
        data (pd.DataFrame): Pandas dataframe containing prediction data
        model (_type_): RandomForestRegressor
        columns_key (list): columns that uniquely identify a record
        columns_data (list): list of column names used for training
        columns_labels (list): list of column names to predict

    Returns:
        pd.DataFrame: Dataframe containing the original data and the MLP gap filled data.
    """
    # identify rows where labels are NaN, but features aren't
    mask_nan = data[columns_labels].isna().any(axis=1)
    mask_not_nan = data[columns_data].notna().all(axis=1)

    # Combine the masks
    combined_mask = mask_nan & mask_not_nan

    # data used for prediction
    X = data[combined_mask][columns_data].reset_index(drop=True)

    # create dataframe of predictions 
    columns_labels_pred = [col.replace('_orig', '') + '_f_rf'  for col in columns_labels]
    y = model.predict(X)
    y = pd.DataFrame(y, columns=columns_labels_pred)
    # merge predictions onto features
    data_pred = pd.concat([X, y], axis=1)


    # merge both dataframes
    data_merged = data.merge(data_pred[columns_key + columns_labels_pred], how="outer", on=columns_key)

    # now, the gapfilled columns have nan values where the original data is not nan. In this case, just take the original values
    for col_f, col in zip(columns_labels_pred, columns_labels):
        data_merged[col_f] = data_merged[col_f].fillna(data_merged[col])

    return data_merged


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