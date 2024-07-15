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
from modules.dataset_util import SingleBatchDataLoader, SplitTimeSeries, TimeSeries_SplitLoader, grab_filled_data
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

FEATURES = COLS_IMPORTANT_FEATURES
series_data, series_targets = grab_filled_data(features=FEATURES)
    
trainloader, valloader, testloader = TimeSeries_SplitLoader(series_data=series_data, series_targets=series_targets)

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