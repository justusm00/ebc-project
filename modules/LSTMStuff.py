import os
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import fastprogress
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader
from modules.dataset_util import SplittedTimeSeries
from modules.paths import PATH_MODEL_TRAINING, PATH_PREPROCESSED
from modules.MLPstuff import EarlyStopper
from modules.util import transform_timestamp
from sklearn.preprocessing import StandardScaler

#### SEEDING ####
SEED = 42
np.random.seed(SEED)
torch.manual_seed(42)
# If using CUDA (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class ConvNet(nn.Module):
    def __init__(self, num_features, window_size, act_fn=nn.ReLU(), kernel_size=6, out_channels=16):
        super().__init__()

        self.conv1 = nn.Conv1d( in_channels=num_features, out_channels=out_channels, kernel_size = kernel_size ) # Output is 16*37
        self.conv2 = nn.Conv1d( in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        self.fc1 = nn.Linear( 896, 80) # 64 Output channels with a size of (ws-12-6-2)
        self.fc2 = nn.Linear( 80, 80)
        self.fc3 = nn.Linear( 80, 2)

        self.MP1 = nn.MaxPool1d( kernel_size=2)

        self.dropout = nn.Dropout(0.2)

        self.act_fn = act_fn

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.act_fn( self.conv1(x) )
        x = self.MP1( self.act_fn( self.conv2(x) ) )
        #print(x.size())
        x = torch.flatten(x, 1) # Flatten for FC
        x = self.act_fn(self.dropout(self.fc1(x)))
        x = self.act_fn(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer with dropout between layers (if num_layers > 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        # Dropout after LSTM output before FC layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if bidirectional==True:
            self.fc = nn.Linear(hidden_size*2, output_size)
            self.fc2 = nn.Linear(hidden_size*2, hidden_size*2)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Apply dropout to the last hidden state (after LSTM)
        out = self.dropout(out[:, -1, :])  # Use the last time step's output for prediction

        # Fully connected layer
        #out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out


# LSTM CNN hybrid model
class LSTMCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=False, cnn_out_channels=64, kernel_size=3):
        super(LSTMCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # CNN Layer (1D Convolution for time series data)
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=1)
        self.cnn_pool = nn.MaxPool1d(kernel_size=2)  # Optional: Use max pooling to reduce dimensionality

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        # Dropout after LSTM and CNN output before fully connected layers
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        combined_input_size = cnn_out_channels + lstm_output_size  # CNN + LSTM output sizes

        self.fc1 = nn.Linear(combined_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(combined_input_size, output_size)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN branch
        x_cnn = x.permute(0, 2, 1)  # Change to [batch_size, features, seq_length] for Conv1D
        x_cnn = self.cnn(x_cnn)  # Apply convolution
        x_cnn = self.relu(x_cnn)
        x_cnn = self.cnn_pool(x_cnn)  # Apply pooling (optional)
        x_cnn = x_cnn.mean(dim=-1)  # Global average pooling (optional)

        # LSTM branch
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        x_lstm, _ = self.lstm(x, (h0, c0))
        x_lstm = self.dropout(x_lstm[:, -1, :])  # Take output from last time step

        # Concatenate CNN and LSTM outputs
        x_combined = torch.cat((x_cnn, x_lstm), dim=1)

        # Fully connected layers
        # x_combined = self.relu(self.fc1(x_combined))
        # x_combined = self.relu(self.fc2(x_combined))
        
        # Final output layer
        out = self.fc_out(x_combined)
        return out
    
    # LSTM CNN hybrid model
class LSTMCNN_maxed(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=False, cnn_out_channels=64, kernel_size=3):
        super(LSTMCNN_maxed, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # CNN Layer (1D Convolution for time series data)
        self.conv1 = nn.Conv1d( in_channels=input_size, out_channels=cnn_out_channels, kernel_size = kernel_size ) # Output is 16*37
        self.conv2 = nn.Conv1d( in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=kernel_size)

        self.MP1 = nn.MaxPool1d( kernel_size=2)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        # Dropout after LSTM and CNN output before fully connected layers
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        combined_input_size = cnn_out_channels + lstm_output_size  # CNN + LSTM output sizes

        self.fc1 = nn.Linear( combined_input_size, 80) # 64 Output channels with a size of (ws-12-6-2)
        self.fc2 = nn.Linear( 80, 80)
        self.fc3 = nn.Linear( 80, 2)
        #self.fc_out = nn.Linear(64, output_size)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN branch
        x_cnn = x.permute(0, 2, 1)  # Change to [batch_size, features, seq_length] for Conv1D
        x_cnn = self.relu( self.conv1(x_cnn) )
        x_cnn = self.MP1( self.relu( self.conv2(x_cnn) ) )
        x_cnn = torch.flatten(x_cnn, 1)

        # LSTM branch
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        x_lstm, _ = self.lstm(x, (h0, c0))
        x_lstm = self.dropout(x_lstm[:, -1, :])  # Take output from last time step

        # Concatenate CNN and LSTM outputs
        x_combined = torch.cat((x_cnn, x_lstm), dim=1)

        # Fully connected layers
        # x_combined = self.relu(self.fc1(x_combined))
        # x_combined = self.relu(self.fc2(x_combined))
        
        # Final output layer
        out = self.relu(self.dropout(self.fc1(x_combined)))
        out = self.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out



def train(dataloader, optimizer, model, master_bar, device, loss_fn = nn.MSELoss()):
    """Run one training epoch.

    Args:
        dataloade: dataloader containing trainingdata
        optimizer: Torch optimizer object
        model: the model that is trained
        loss_fn: the loss function to be used -> nn.MSELoss()
        master_bar: Will be iterated over for each
            epoch to draw batches and display training progress

    Returns:
        Mean epoch loss and accuracy
    """
    losses = []  # Use a list to store individual batch losses

    for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
        optimizer.zero_grad()
        model.train()

        # Forward pass
        y_pred = model(x.to(device, non_blocking=True))

        # Compute loss
        batch_loss = loss_fn(y_pred, y.to(device, non_blocking=True))

        # Backward pass
        batch_loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Save the batch loss for logging purposes
        losses.append(batch_loss.item())

    # Calculate the mean loss for the epoch
    mean_loss = np.mean(losses)

    # Return the mean loss for the epoch
    return mean_loss





def validate(dataloader, model, master_bar, device, loss_fn=nn.MSELoss()):
    """Compute loss and total prediction error on validation set.

    Args:
        dataloader: dataloader containing validation data
        model (nn.Module): the model to train
        loss_fn: the loss function to be used, defaults to MSELoss
        master_bar (fastprogress.master_bar): Will be iterated over to draw 
            batches and show validation progress

    Returns:
        Mean loss and total prediction error on validation set
    """
    epoch_loss = []

    model.eval()
    with torch.no_grad():
        for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
            # make a prediction on validation set
            y_pred = model(x.to(device, non_blocking=True))

            # Compute loss
            loss = loss_fn(y_pred, y.to(device, non_blocking=True))

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss)


def test(dataloader, model, device, loss_fn=nn.MSELoss()):
    """Compute loss on testset.

    Args:
        dataloader: dataloader containing validation data
        model (nn.Module): the model to train
        loss_fn: the loss function to be used, defaults to MSELoss

    Returns:
        Mean loss 
    """
    epoch_loss = []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # make a prediction on test set
            y_pred = model(x.to(device, non_blocking=True))

            # Compute loss
            loss = loss_fn(y_pred, y.to(device, non_blocking=True))

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss)



def plot(title, label, train_results, val_results, yscale='linear', save_path=None):
    """Plot learning curves.

    Args:
        title: Title of plot
        label: y-axis label
        train_results: Vector containing training results over epochs
        val_results: vector containing validation results over epochs
        yscale: Defines how the y-axis scales
        save_path: Optional path for saving file
    """
    
    epochs = np.arange(len(train_results)) + 1
    
    sns.set(style='ticks')

    plt.plot(epochs, train_results, epochs, val_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Validation results']
        
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)
    
    sns.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()







def run_training(model, optimizer, num_epochs, train_dataloader, val_dataloader, device, 
                 loss_fn=nn.MSELoss(), patience=1, early_stopper=None, scheduler=None, verbose=False, plot_results=True, save_plots_path=None):
    """Run model training.

    Args:
        model: The model to be trained
        optimizer: The optimizer used during training
        loss_fn: Torch loss function for training -> nn.MSELoss()
        num_epochs: How many epochs the model is trained for
        train_dataloader:  dataloader containing training data
        val_dataloader: dataloader containing validation data
        verbose: Whether to print information on training progress

    Returns:
        lists containing  losses and total prediction errors per epoch for training and validation
    """
    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses = [],[]

    if early_stopper:
        ES = EarlyStopper(verbose=verbose, patience = patience)

    # initialize old loss value varibale (choose something very large)
    val_loss_old = 1e6

    for epoch in master_bar:
        # Train the model
        epoch_train_loss = train(dataloader=train_dataloader, optimizer=optimizer, model=model, 
                                                 master_bar=master_bar, device=device, loss_fn=loss_fn)
        # Validate the model
        epoch_val_loss = validate(val_dataloader, model, master_bar, device, loss_fn)

        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}')


        if early_stopper and epoch != 0:
            ES.check_criterion(epoch_val_loss, val_loss_old)
            if ES.early_stop:
                master_bar.write("Early stopping")
                model = ES.load_checkpoint()
                break
            if ES.counter > 0:
                master_bar.write(f"Early stop counter: {ES.counter} / {patience}")

        # Save smallest loss
        if early_stopper and epoch_val_loss < val_loss_old:
            val_loss_old = epoch_val_loss
            ES.save_model(model)
            
        if scheduler:
            scheduler.step(epoch_val_loss)

    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')

    if plot_results:
        plot("Loss", "Loss", train_losses, val_losses, save_path=save_plots_path)
    return train_losses, val_losses





def extract_subseries(df, window_size, features, labels, index=0, n_aug=0):
    subseries = []

    for i in range(len(df) - window_size + 1):  # Go over all possible series
        window = df.iloc[i:i+window_size].copy(deep=True)  # Extract window through slicing

        # Make sure that none of the important features of the window contain NaNs
        if window[features + labels].isna().sum().sum() == 0:
            window_selected = window[features + labels].copy(deep=True) # Extract selected features and labels
            window_selected['series_id'] = index
            subseries.append( window_selected.reset_index(drop=True) )
            index += 1

            # Now create the augmentations by adding Gaussian noise to them
            for _ in range(n_aug):
                window_aug = window_selected.copy(deep=True)

                for lab in labels:
                    noise = np.random.normal(0, 1, size=window_size)
                    window_aug.loc[:, lab] += noise  # Use .loc to avoid SettingWithCopyWarning

                window_aug['series_id'] = index
                subseries.append( window_aug.reset_index(drop=True) )
                index += 1

    # Convert the list of subseries into a DataFrame and return
    return pd.concat(subseries, ignore_index=True), index

def grab_series_data(window_size, features, labels, n_aug=0, gapfilled=False, normalize=False):
    data = pd.read_csv('data/preprocessed/data_merged_with_nans.csv')

    def convert_to_timestamp(df):
        # Combine year and DOY (day of year)
        df['datetime'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j')
        # Add the 30min interval as hours and minutes
        df['time_delta'] = pd.to_timedelta(df['30min'] * 30, unit='m')
        df['timestamp'] = df['datetime'] + df['time_delta']
        #print(df['timestamp'])
        return df

    if gapfilled:
        # First only take in the data in time of the recordings
        BG = pd.read_csv('data/gapfilled/BG_gapfilled.csv')
        GW = pd.read_csv('data/gapfilled/GW_gapfilled.csv')
        # Filter out time that wasn't measured. The division into
        # subframes is important for no subseries overlapping between
        # recorded times
        BG = convert_to_timestamp(BG)
        GW = convert_to_timestamp(GW)

        start_time = pd.to_datetime('2023-08-01 00:00:00')
        end_time = pd.to_datetime('2024-04-01 00:00:00')
        BG['filter_time'] = pd.to_datetime(BG['timestamp'])
        GW['filter_time'] = pd.to_datetime(GW['timestamp'])
        BG_23 = BG[BG['filter_time'] < start_time].copy(deep=True)
        GW_23 = GW[GW['filter_time'] < start_time].copy(deep=True)
        BG_24 = BG[BG['filter_time'] > end_time].copy(deep=True)
        GW_24 = GW[GW['filter_time'] > end_time].copy(deep=True)

    else:
        # Split the data based on location (0 for GW, 1 for BG)
        GW_data = data[data['location'] == 0].copy(deep=True)
        BG_data = data[data['location'] == 1].copy(deep=True)

        # Define the start and end times (for exclusion)
        start_time = pd.to_datetime('2023-08-01')
        end_time = pd.to_datetime('2024-04-01')

        # Convert year and DOY to a datetime column in both datasets
        GW_data = convert_to_timestamp(GW_data)
        BG_data = convert_to_timestamp(BG_data)


        # Filter out the rows between the start and end times
        GW_23 = GW_data[(GW_data['timestamp'] < start_time)].copy(deep=True)
        GW_24 = GW_data[(GW_data['timestamp'] > end_time)].copy(deep=True)
        
        BG_23 = BG_data[(BG_data['timestamp'] < start_time)].copy(deep=True)
        BG_24 = BG_data[(BG_data['timestamp'] > end_time)].copy(deep=True)

    print(f"BG23: {len(BG_23)}, GW_24: {len(GW_23)}, BG24: {len(BG_24)}, GW24: {len(GW_24)}")

    # Transform timestamp
    BG_23 = transform_timestamp(BG_23, col_name='timestamp')
    GW_23 = transform_timestamp(GW_23, col_name='timestamp')
    BG_24 = transform_timestamp(BG_24, col_name='timestamp')
    GW_24 = transform_timestamp(GW_24, col_name='timestamp')

    # Encode location
    BG_23['location'] = 0
    GW_23['location'] = 1
    BG_24['location'] = 0
    GW_24['location'] = 1

    if normalize:
        feats_BG_23 = BG_23[features].copy(deep=True)
        feats_GW_23 = GW_23[features].copy(deep=True)
        feats_BG_24 = BG_24[features].copy(deep=True)
        feats_GW_24 = GW_24[features].copy(deep=True)

        scaler = StandardScaler()

        feats_BG_23 = scaler.fit_transform(feats_BG_23)
        feats_GW_23 = scaler.transform(feats_GW_23)
        feats_BG_24 = scaler.transform(feats_BG_24)
        feats_GW_24 = scaler.transform(feats_GW_24)

        BG_23.loc[:,features] = feats_BG_23
        GW_23.loc[:,features] = feats_GW_23
        BG_24.loc[:,features] = feats_BG_24
        GW_24.loc[:,features] = feats_GW_24

    # Now extract the subseries for the frames
    BG_23_ss, index = extract_subseries(BG_23, window_size=window_size, features=features, labels=labels, index=0, n_aug=n_aug)
    print("BG_23 done")
    BG_24_ss, index = extract_subseries(BG_24, window_size=window_size, features=features, labels=labels, index=index, n_aug=n_aug)
    print("BG_24 done")
    GW_23_ss, index = extract_subseries(GW_23, window_size=window_size, features=features, labels=labels, index=index, n_aug=n_aug)
    print("GW_23 done")
    GW_24_ss, index = extract_subseries(GW_24, window_size=window_size, features=features, labels=labels, index=index, n_aug=n_aug)
    print("GW_24 done")

    # Concatenate for one dataframe
    subseries = pd.concat([BG_23_ss, BG_24_ss, GW_23_ss, GW_24_ss], ignore_index=True)
    print(f"{ (1+n_aug) * index } subseries have been found, of which {int(np.ceil(n_aug*n_aug/(n_aug+1) * index ))} are augmmentations of original data")
    return subseries




def normalize_features(train_data, val_data, test_data, features):
    # Extract the feature columns that need to be normalized
    train_features = train_data[features].copy(deep=True)
    val_features = val_data[features].copy(deep=True)
    test_features = test_data[features].copy(deep=True)
    
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit on the training data and transform it
    train_features_normalized = scaler.fit_transform(train_features)

    # Use the same scaler (fitted on training data) to transform validation and test data
    val_features_normalized = scaler.transform(val_features)
    test_features_normalized = scaler.transform(test_features)

    # Replace the feature columns in the original data with the normalized values
    train_data.loc[:, features] = train_features_normalized
    val_data.loc[:, features] = val_features_normalized
    test_data.loc[:, features] = test_features_normalized

    return train_data, val_data, test_data

# Create split and data loaders
def Splitted_SplitLoader(series_data, series_targets, batch_size, features, normalize=True):
    # Create array of all possible indices
    idxs = series_data[ 'series_id' ].unique()

    test_split_size = 0.9
    val_split_size = 0.9

    batch_size = batch_size

    np.random.shuffle( idxs ) # Shuffle the indexes randomly

    # Train test split
    splt = int( test_split_size * len(idxs) )
    train_idxs = idxs[:splt]
    test_idxs = idxs[splt:]

    # Now train val split
    splt = int( val_split_size * len(train_idxs) )
    val_idxs = train_idxs[splt:]
    train_idxs = train_idxs[:splt]

    # Assign new sets
    data_train = series_data[ series_data['series_id'].isin( train_idxs ) ]
    data_val = series_data[ series_data['series_id'].isin( val_idxs ) ]
    data_test = series_data[ series_data['series_id'].isin( test_idxs ) ]

    labels_train = series_targets[ series_data['series_id'].isin( train_idxs ) ]
    labels_val = series_targets[ series_data['series_id'].isin( val_idxs ) ]
    labels_test = series_targets[ series_data['series_id'].isin( test_idxs ) ]

    if normalize:
        data_train, data_val, data_test = normalize_features(data_train, data_val, data_test, features)

    trainset = SplittedTimeSeries(data_train, labels_train)
    valset = SplittedTimeSeries(data_val, labels_val)
    testset = SplittedTimeSeries(data_test, labels_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=0, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=True)

    return trainloader, valloader, testloader