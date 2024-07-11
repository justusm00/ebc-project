import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from modules.MLPstuff import run_training
from modules.columns import COLS_IMPORTANT_FEATURES, COLS_LABELS_ALL

# Get number of cpus to use for faster parallelized data loading
avb_cpus = os.cpu_count()
num_cpus = int(3 * avb_cpus / 4)
print(avb_cpus, 'CPUs available,', num_cpus, 'were assigned' )

def get_device(cuda_preference=True):
        """Gets pytorch device object. If cuda_preference=True and 
            cuda is available on your system, returns a cuda device.
        
        Args:
            cuda_preference: bool, default True
                Set to true if you would like to get a cuda device
                
        Returns: pytorch device object
                Pytorch device
        """
        
        print('cuda available:', torch.cuda.is_available(), 
            '; cudnn available:', torch.backends.cudnn.is_available(),
            '; num devices:', torch.cuda.device_count())
        
        use_cuda = False if not cuda_preference else torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
        print(f'Using device {device_name} \n')
        return device

device = get_device()

# Load the data
series = pd.read_csv('data/preprocessed/data_merged_with_nans.csv').dropna(subset=["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit", "H_f", "LE_f"])
series = series[COLS_IMPORTANT_FEATURES + ["H_f", "LE_f"]] #series[ COLS_IMPORTANT_FEATURES + COLS_LABELS_ALL ]
# Check for no NaNs in the used columns
print("NaNs in columns:")
print(series.isna().sum())
# Split into targets and features
series_data = series[COLS_IMPORTANT_FEATURES]
series_targets = series[["H_f", "LE_f"]]

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
            target = self.series_targets[ i+self.window_size ]

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
    
# Create the dataset and do the train test val split
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
window_size = 48 # Use one day
batch_size = 64

trainset = SplitTimeSeries(data_train, targets_train, window_size=window_size)
testset = SplitTimeSeries(data_test, targets_test, window_size=window_size)
valset = SplitTimeSeries(data_val, targets_val, window_size=window_size)

# num_workers=0 because it strangely doesn't work otherwise
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0)
valloader = DataLoader(valset, batch_size=batch_size, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, num_workers=0)

iterator = iter(trainloader)
dat, tar = next(iterator)
print(dat.size())
print(tar.size())

# Define the model.
# The idea is to have 1d convolutions for simple time series analyses and 2d convolutions downstream to capture feature interactions
class ConvNet(nn.Module):
    def __init__(self, num_features, window_size, act_fn=nn.ReLU()):
        super().__init__()

        self.conv1 = nn.Conv1d( in_channels=num_features, out_channels=16, kernel_size = 12 ) # Output is 16*37
        self.conv2 = nn.Conv1d( in_channels=16, out_channels=32, kernel_size=6 ) # Output size 32*32 (32 series length)
        self.conv3 = nn.Conv2d( in_channels=1, out_channels=64, kernel_size=(3,32) ) # Convolve over all features # Ouput size is 64*1*30

        self.fc1 = nn.Linear( 64 * (window_size-18) * 1, 2) # 64 Output channels with a size of (ws-12-6-2)

        self.act_fn = act_fn

    def forward(self, x):
        x = self.act_fn( self.conv1(x) ) 
        x = self.act_fn( self.conv2(x) )
        x = x.unsqueeze(1) # Add dimension above for 2DConv
        x = self.act_fn( self.conv3(x) )
        x = torch.flatten(x, 1) # Flatten for FC
        x = self.fc1(x)
        return x
    
model = ConvNet(num_features=len(COLS_IMPORTANT_FEATURES), window_size=window_size).to(device)

criterion = nn.MSELoss()
lr = 10**(-2)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8,
                              threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0)

num_epochs = 50

train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=num_epochs, train_dataloader=trainloader, val_dataloader=valloader, 
                                                              device=device, loss_fn=criterion, patience=10, scheduler=scheduler, early_stopper=False, verbose=True)