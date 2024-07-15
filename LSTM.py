import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from modules.util import transform_timestamp
from modules.MLPstuff import run_training, MyReduceLROnPlateau
from modules.columns import COLS_IMPORTANT_FEATURES, COLS_LABELS_ALL, COLS_KEY, COLS_KEY_ALT
from modules.dataset_util import SingleBatchDataLoader
from modules.paths import PATH_MODEL_SAVES_CONV, PATH_MODEL_SAVES_LSTM

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
FEATURES = COLS_IMPORTANT_FEATURES
series = series.dropna(subset=FEATURES+['H_f_mlp','LE_f_mlp'])
series_data = series[COLS_IMPORTANT_FEATURES]
# FEATURES = COLS_KEY_ALT + ["incomingShortwaveRadiation"]
# series = series.dropna(subset=FEATURES + ['H_f_mlp','LE_f_mlp'])
# series_data = series[FEATURES]
series_targets = series[["H_f_mlp", "LE_f_mlp"]]

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
window_size = 24 # Use one day
batch_size = 64

trainset = SplitTimeSeries(data_train, targets_train, window_size=window_size)
testset = SplitTimeSeries(data_test, targets_test, window_size=window_size)
valset = SplitTimeSeries(data_val, targets_val, window_size=window_size)

# num_workers=0 because it strangely doesn't work otherwise
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0)
valloader = DataLoader(valset, batch_size=batch_size, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, num_workers=0)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.transpose(1,2) # Input Features are supposed to be in the last dimension
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the last time step's output for prediction
        return out
    
# Try to overfit first
trainloader_of = SingleBatchDataLoader(trainloader)
valloader_of = SingleBatchDataLoader(valloader)

#model = ConvNet(num_features=len(FEATURES), window_size=window_size).to(device)
model = LSTMModel(input_size=len(FEATURES), hidden_size=40, num_layers=1, output_size=2).to(device)
# Set loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=3000,
                                            train_dataloader=trainloader_of, val_dataloader=valloader_of,
                                            device=device, loss_fn=criterion, patience=5, early_stopper=False, verbose=False, plot_results=True)

print(f"Overfitting: train loss: {train_losses[-1]}, val loss: {val_losses[-1]}")

#model = ConvNet(num_features=len(FEATURES), window_size=window_size).to(device)
hidden_size=80
num_layers=2

model = LSTMModel(input_size=len(FEATURES), hidden_size=hidden_size, num_layers=num_layers, output_size=2).to(device)

criterion = nn.MSELoss()
lr = 10**(-3)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = MyReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5)
#scheduler = None

num_epochs = 200

train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=num_epochs, train_dataloader=trainloader, val_dataloader=valloader, 
                                                              device=device, loss_fn=criterion, patience=20, scheduler=scheduler, early_stopper=True, verbose=False)

print(f"Train loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")

# Save model
model_save_path = PATH_MODEL_SAVES_LSTM + f"LSTM_{hidden_size}_{num_layers}_IF_WS{window_size}" + '.pth'
torch.save(model.state_dict(), model_save_path )
print(f"Saved model to {model_save_path} \n")