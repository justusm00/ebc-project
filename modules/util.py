import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import fastprogress


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
   
    # make sure there are not duplicates
    columns_data = list(set(columns_data))
    
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



class EarlyStopper:
    """Early stops the training if validation accuracy does not increase after a
    given patience. Saves and loads model checkpoints.
    """
    def __init__(self, verbose=False, path='checkpoint.pt', patience=1):
        """Initialization.

        Args:
            verbose (bool, optional): Print additional information. Defaults to False.
            path (str, optional): Path where checkpoints should be saved. 
                Defaults to 'checkpoint.pt'.
            patience (int, optional): Number of epochs to wait for increasing
                accuracy. If accuracy does not increase, stop training early. 
                Defaults to 1.
        """
        ####################
        self.verbose = verbose
        self.path = path
        self.patience = patience
        self.counter = 0
        ####################

    @property
    def early_stop(self):
        """True if early stopping criterion is reached.

        Returns:
            [bool]: True if early stopping criterion is reached.
        """
        if self.counter == self.patience:
            return True

    def save_model(self, model):
        # scripted save
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(self.path)
        return
        
    def check_criterion(self, loss_val_new, loss_val_old):
        if loss_val_old <= loss_val_new:
            self.counter += 1
        else:
            self.counter = 0
        
        return
    
    def load_checkpoint(self):
        model = torch.jit.load(self.path)
        return model
    # define more methods required to make `EarlyStopper` functional