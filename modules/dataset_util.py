from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.paths import PATH_MODEL_TRAINING, PATH_PREPROCESSED
from modules.util import transform_timestamp
import pickle



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
    


# Define a dataset of windows
class SplitTimeSeries(Dataset):
    def __init__(self, series_data, series_targets, window_size=48):
        self.series_data = series_data.to_numpy()
        self.series_targets = series_targets.to_numpy()
        self.window_size = window_size
        self.splits, self.targets = self.split_series()

    def split_series(self):
        splits = []
        targets = []

        # Splits of size windows_size result in a loss of windows_size data points
        for i in range(len(self.series_data) - self.window_size): 
            # Create split
            split = self.series_data[i:i+self.window_size, :] # Slice through series
            # Target are the target values at the end of the snippet
            target = self.series_targets[ i+self.window_size-1 ]

            splits.append(split)
            targets.append(target)

        return np.array(splits), np.array(targets) # Convert to numpy for faster processing
    
    def __len__(self):
        return len(self.splits)
    
    def __getitem__(self, idx):
        # Return torch tensors of the splits
        tens_dat = torch.tensor( self.splits[idx], dtype=torch.float32 )
        tens_tar = torch.tensor( self.targets[idx], dtype=torch.float32 )
        return torch.transpose(tens_dat, 0, 1), tens_tar
    


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
def grab_data(model_hash, num_cpus, fill_artificial_gaps=False, cols_features=None, cols_labels=None, normalization=False, minmax_scaling=False):
    """Loads training and test data from respective directories. 

    Args:
        path_data (_type_): _description_
        path_indices 
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


    train_indices, test_indices = get_train_test_indices(fill_artificial_gaps, model_hash)

    path_data = PATH_PREPROCESSED + 'data_merged_with_nans.csv'


    data = pd.read_csv(path_data)

    trainset = data.loc[train_indices]
    testset = data.loc[test_indices]

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



def train_test_splitter(df, cols_features, cols_labels, model_hash, fill_artificial_gaps=True, path_save=None, test_size=0.2, 
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
    if verbose:
        print(f"Number of records in original data: {df.shape[0]}")


    if fill_artificial_gaps:
        # drop nan values
        df = df[cols_features + cols_labels]
        df = df.dropna()        
        
        # Split the data
        train_indices, test_indices = train_test_split(df.index, test_size=test_size, random_state=random_state)


    else:
        df_train = df[df["artificial_gap"] == 0]
        df_test = df[df["artificial_gap"] != 0]

        # drop nan values
        df_train = df_train[cols_features + cols_labels]
        df_test = df_test[cols_features + cols_labels]

        df_train = df_train.dropna()
        df_test = df_test.dropna()

        train_indices = df_train.index
        test_indices = df_test.index


    if verbose:
        print(f"Number of train values: {len(train_indices)}")
        print(f"Number of test values: {len(test_indices)}")


    # Data to be saved
    indices = {
        'train_indices': train_indices,
        'test_indices': test_indices
    }


    
    if path_save:
        # Save to a file
        if fill_artificial_gaps:
            file_path = path_save + 'indices_' + model_hash + '.pkl'
        else:
            file_path = path_save + 'indices_AGF_' + model_hash + '.pkl'

        with open(file_path, 'wb') as file:
            pickle.dump(indices, file)
        if verbose:
            print(f"Saved train data to {file_path} \n") 

    return train_indices, test_indices




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


def TimeSeries_SplitLoader(series_data, series_targets, window_size):
    test_split_size = 0.9
    val_split_size = 0.8

    # Start wiht train test split
    idxs = list( range( len(series_data) ) )
    splt = int( test_split_size * len(series_data) )
    np.random.shuffle( idxs ) # Shuffle the indexes randomly
    train_idxs = idxs[:splt]
    test_idxs = idxs[splt:]

    # Now train val split
    splt = int( val_split_size * len(train_idxs) )
    val_idxs = train_idxs[splt:]
    train_idxs = train_idxs[:splt]

    # Now perform the split
    data_train = series_data.iloc[train_idxs]
    data_val = series_data.iloc[val_idxs]
    data_test = series_data.iloc[test_idxs]

    targets_train = series_targets.iloc[train_idxs]
    targets_val = series_targets.iloc[val_idxs]
    targets_test = series_targets.iloc[test_idxs]

    # Normalize the data using the statistics of the training set (avoid spilling)
    data_mean = data_train.mean()
    data_std = data_train.std()

    data_train = (data_train - data_mean) / data_std
    data_test = (data_test - data_mean) / data_std
    data_val = (data_val - data_mean) / data_std

    print(f"The data has {len(series_data)} entries.\nThe trainset has {len(data_train)}, the valset {len(data_val)} and the testset {len(data_test)} entries.")

    # Create the datasets and dataloaders
    window_size = 24 # Use one day
    batch_size = 64

    trainset = SplitTimeSeries(data_train, targets_train, window_size=window_size)
    testset = SplitTimeSeries(data_test, targets_test, window_size=window_size)
    valset = SplitTimeSeries(data_val, targets_val, window_size=window_size)

    # num_workers=0 because it strangely doesn't work otherwise
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=0)

    return trainloader, valloader, testloader

def grab_filled_data(features):
    BG = pd.read_csv('data/gapfilled/BG_gapfilled.csv')
    GW = pd.read_csv('data/gapfilled/GW_gapfilled.csv')
    # Filter out time that wasn't measured
    start_time = pd.to_datetime('2023-08-01 00:00:00')
    end_time = pd.to_datetime('2024-04-01 00:00:00')
    BG['filter_time'] = pd.to_datetime(BG['timestamp'])
    GW['filter_time'] = pd.to_datetime(GW['timestamp'])
    BG = BG[ (BG['filter_time'] < start_time) | (BG['filter_time'] > end_time) ]
    GW = GW[ (GW['filter_time'] < start_time) | (GW['filter_time'] > end_time) ]

    # Transform timestep
    BG = transform_timestamp(BG, col_name='timestamp')
    GW = transform_timestamp(GW, col_name='timestamp')
    # Encode location
    BG['location'] = 0
    GW['location'] = 1
    # Concat for resulting timeseries
    series = pd.concat([BG, GW])

    # series = pd.read_csv('data/gapfilled/data_merged_with_nans.csv').dropna(subset=["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit", "H_f", "LE_f"])
    # series = series[COLS_IMPORTANT_FEATURES + ["H_f", "LE_f"]] #series[ COLS_IMPORTANT_FEATURES + COLS_LABELS_ALL ]
    # Check for no NaNs in the used columns
    print("NaNs in columns:")
    print(series.isna().sum())
    # Split into targets and features
    
    series = series.dropna(subset=features+['H_f_mlp','LE_f_mlp'])
    series_data = series[features]
    # FEATURES = COLS_KEY_ALT + ["incomingShortwaveRadiation"]
    # series = series.dropna(subset=FEATURES + ['H_f_mlp','LE_f_mlp'])
    # series_data = series[FEATURES]
    series_targets = series[["H_f_mlp", "LE_f_mlp"]]

    return series_data, series_targets



def get_train_test_indices(fill_artificial_gaps, model_hash):
    # determine path to train / test indices
    if fill_artificial_gaps:
        path_indices = PATH_MODEL_TRAINING + 'indices_AGF_' + model_hash + '.pkl'
    else:
        path_indices = PATH_MODEL_TRAINING + 'indices_' + model_hash + '.pkl'
    # load data
    # Load from the file
    with open(path_indices, 'rb') as file:
        indices = pickle.load(file)

    train_indices = indices['train_indices']
    test_indices = indices['test_indices']

    return train_indices, test_indices