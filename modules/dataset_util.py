from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# Define Dataset
class EBCDataset(Dataset):
    def __init__(self, data, labels, normalization=False, minmax_scaling=False, means=None, stds=None, mins=None, maxs=None):
        if (normalization is True ) and ( means is None or stds is None) :
            raise ValueError("Must specify mean and standard for normalization.")
        if (minmax_scaling is True ) and ( mins is None or maxs is None) :
            raise ValueError("Must specify min and max for min_max_scaling.")
        if (minmax_scaling is True ) and (normalization is True ) :
            raise ValueError("Can only perform normalization OR minmax_scaling")
        self.data = data
        self.labels = labels
        self.normalization = normalization
        self.minmax_scaling = minmax_scaling
        self.means = means
        self.stds = stds
        self.mins = mins
        self.maxs = maxs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.normalization:
            sample = (sample - self.means) / self.stds 
        if self.minmax_scaling:
            sample = (sample - self.mins) / (self.maxs - self.mins)
        return sample, label
    


# Function to create a sub-dataloader that only iterates over the first batch
class SingleBatchDataLoader:
    def __init__(self, dataloader):
        self.batch = next(iter(dataloader))
        
    def __iter__(self):
        return iter([self.batch])
    
    def __len__(self):
        return 1


############# DATASET FUNCTIONS #############

# Data loader
def grab_data(path_train, path_test, num_cpus, cols_features=None, cols_labels=None, normalization=False, minmax_scaling=False):
    """Loads training and test data from respective directories. 

    Args:
        path_train (_type_): _description_
        path_test (_type_): _description_
        num_cpus (_type_): _description_
        cols_features (_type_, optional): _description_. Defaults to None.
        cols_labels (_type_, optional): _description_. Defaults to None.
        normalization (bool, optional): If True, normalize data based on trainset statistics.
        minmax_scaling (bool, optional): If True, perform minmax scaling


    Returns:
        _type_: _description_
    """
    if (minmax_scaling is True ) and (normalization is True ) :
        raise ValueError("Can only perform normalization OR minmax_scaling")

    # load data
    trainset = pd.read_csv(path_train)
    testset = pd.read_csv(path_test)

    # Select data and labels
    if cols_features == None:
        cols_features = ['CO2', 'H2O', 'Ustar', 'location', 'year', 'month', 'day', '30min']
    if cols_labels == None:
        cols_labels = ['H_orig', 'LE_orig']
   

    # Convert to torch tensor
    trainset_data = torch.tensor(trainset[ cols_features ].values, dtype=torch.float32)
    trainset_labels = torch.tensor(trainset[ cols_labels].values, dtype=torch.float32)
    testset_data = torch.tensor(testset[ cols_features ].values, dtype=torch.float32)
    testset_labels = torch.tensor(testset[ cols_labels].values, dtype=torch.float32)
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

    if minmax_scaling: # normalize to [-1,1]
        num_samples = trainset.data.shape[0]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_samples, 
                                                    num_workers=num_cpus)
        
        # compute column means and stds
        rows, _ = next(iter(trainloader))
        trainset_min, _ = torch.min(rows, dim=(0)) 
        trainset_max, _ = torch.max(rows, dim=(0))

        # load again, now normalized
        trainset = EBCDataset(trainset_data, trainset_labels, minmax_scaling=True, mins=trainset_min, maxs=trainset_max)
        testset = EBCDataset(testset_data, testset_labels, minmax_scaling=True, mins=trainset_min, maxs=trainset_max)
        

    return trainset, testset


def train_test_splitter(path_data, cols_features, cols_labels, model_hash, path_save=None, test_size=0.2, 
                           random_state=42, verbose=True):
    """Perform random train test split and drop nan values. This needs to be done for each unique combination of features and labels since the data availability depends on this combination. The train and test data are save to path_save and identified by a unique hash generated from the feature-label-combination.

    Args:
        df (_type_): _description_
        cols_features (_type_): _description_
        cols_labels (_type_): _description_
        path_save (_type_): path where data will be stored. If set to None, don't save datasets and just return them
        model_hash (str): identifying combination of features and labels
        test_size (float, optional): _description_. Defaults to 0.2.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: trainset
        _type_: testset
    """
    # load data
    df = pd.read_csv(path_data)
    if verbose:
        print(f"Number of records in original data: {df.shape[0]}")
    # drop nan values
    df = df[cols_features + cols_labels]
    df = df.dropna()
    if verbose:
        print(f"Number of records after dropping rows with nan values in feature / label columns: {df.shape[0]}")
    # Define features and target
    X = df[cols_features]  # Features
    y = df[cols_labels]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # concatenate again
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)


    if path_save:
        path_train = path_save + 'training_data_' + model_hash + '.csv'
        path_test = path_save + 'test_data_' + model_hash + '.csv'
        # save as csv
        df_train.to_csv(path_train, index=False)
        df_test.to_csv(path_test, index=False)
        if verbose:
            print(f"Saved train data to {path_train} \n")
            print(f"Saved test data to {path_test} \n")    

    return df_train, df_test


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

