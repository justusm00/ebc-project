import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from columns import COLS_FEATURES, COLS_LABELS, COLS_TIME
from modules.MLPstuff import MLP


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
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label



# Data loader
def grab_data(path, columns_data=None, columns_labels=None, num_cpus=1, return_dataset=True):
    """Loads data from data_dir

    Args:
        data_dir (str): Directory to store data
        num_cpus (int, optional): Number of cpus that should be used to 
            preprocess data. Defaults to 1.

    Returns:
        Returns datasets as Dataset class for Göttingen forest and Bothanic Garden combined
        or returns the data and labels as pandas.dataframe for model predictions
    """

    # load data
    data = pd.read_csv(path)

    # Select data and labels
    if columns_data == None:
        columns_data = ['CO2', 'H2O', 'Ustar', 'location', 'year', 'month', 'day', '30min']
    if columns_labels == None:
        columns_labels = ['H_orig', 'LE_orig']
   

    if return_dataset:
        # Convert to torch tensor
        data_tensor = torch.tensor(data[ columns_data ].values, dtype=torch.float32)
        labels_tensor = torch.tensor(data[ columns_labels].values, dtype=torch.float32)

        dataset = EBCDataset(data_tensor, labels_tensor)
        return dataset, len(columns_data), len(columns_labels)
    
    else:
        return data[columns_data], data[columns_labels], len(columns_data), len(columns_labels)



# dataset Splitter 
def train_val_test_splitter(dataset, split_seed=42, test_frac=0.2, val_frac = 0.2):
    """ Splits given dataset into train, val and test datasets

    Args:
        dataset: the given dataset
        split_seed: the seed used for the rng
        test_frac: fraction of data used for testing
        val_frac_ fraction of training data used for validation
    """
    # Train Test Split
    num_test_samples = np.ceil(test_frac * len(dataset)).astype(int)
    num_train_samples = len(dataset) - num_test_samples
    trainset, testset = torch.utils.data.random_split(dataset, 
                                                    (num_train_samples, num_test_samples), 
                                                    generator=torch.Generator().manual_seed(split_seed))
    
    # Train Val Split
    num_val_samples = np.ceil(val_frac * len(trainset)).astype(int)
    num_train_samples = len(trainset) - num_val_samples
    trainset, valset = torch.utils.data.random_split(trainset, 
                                                    (num_train_samples, num_val_samples), 
                                                    generator=torch.Generator().manual_seed(split_seed))
    
    return trainset, valset, testset



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
                                              num_workers=num_cpus)
    valloader = torch.utils.data.DataLoader(valset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_cpus)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True, 
                                             num_workers=num_cpus)
    return trainloader, valloader, testloader



def gap_filling_mlp(path_model, path_data, columns_data, columns_labels):
    """Perform gap filling using pre-trained MLP

    Args:
        path_model (str): path to MLP parameter file
        path_data (str): path to data containing gaps
        columns_data (list of str): list of columns used as features
        columns_labels (list of str): list of columns used as labels 

    Returns:
        Dataframe with gap filled columns in columns_labels. Gap filled columns are called <col_name>_f_mlp.
    """
    input, target, dim_in, dim_out = grab_data(path_data, columns_data=columns_data,
                                               columns_labels=columns_labels, return_dataset = False )
    data = pd.concat([input, target], axis=1)
    # Load the model
    model = MLP(dim_in, dim_out, num_hidden_units=30, num_hidden_layers=4)
    model.load_state_dict(torch.load(path_model))
    # identify rows where labels are NaN, but features aren't
    mask_nan = data[columns_labels].isna().any(axis=1)
    mask_not_nan = data[columns_data].notna().all(axis=1)

    # Combine the masks
    combined_mask = mask_nan & mask_not_nan

    # data used for prediction
    input = data[combined_mask][columns_data].reset_index(drop=True)

    # transform input into torch.tensor and make predictions
    input_tensor = torch.tensor(input.values, dtype=torch.float32)

    with torch.no_grad():
        pred = model(input_tensor).numpy() #  Transform back to numpy 

    # create dataframe of predictions 
    pred = pd.DataFrame(pred, columns=columns_labels)

    # merge predictions onto features
    data_pred = pd.concat([input, pred], axis=1)


    # create original dataframe
    data_orig = data[~mask_nan].reset_index(drop=True)

    # merge both dataframes
    data_merged = pd.concat([data_orig, data_pred])

    # rename columns 
    data_final = data_merged.rename(columns={col: col.replace('_orig', '') + '_f_mlp' for col in columns_labels})

    return data_final
