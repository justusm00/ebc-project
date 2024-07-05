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

# MLP definition
class MLP(nn.Module):
   # Multi layer perceptron torch model
    def __init__(self, num_variables, num_predictors, num_hidden_units=30, 
                 num_hidden_layers=1, act_fn=nn.ReLU()):
        """Initialize model.

        Args:
            num_variables: number of variables used for prediction
            num_predictors: number of variables to be predicted
            num_hidden_units: Hidden units per layers
            num_hidden_layers: Number of hidden layers
            act_fn: Activation function to use after the hidden layers. Defaults to nn.ReLU
        """
        ####################
        
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = [nn.Linear(num_variables, num_hidden_units), act_fn]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_hidden_units, num_hidden_units))
            layers.append(act_fn)
    
        layers.append(nn.Linear(num_hidden_units, num_predictors))
        
        self.linear_relu_stack = nn.Sequential(*layers)
        ####################

    
    def forward(self, x):
        """Compute model predictions.

        Args:
            x: Tensor of data

        Returns:
            tensor of model prediction
        """
        ####################
        x = self.flatten(x)
        pred = self.linear_relu_stack(x)
        return pred
        ####################




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






        

############# TRAINING FUNCTIONS ###############

# Define validation metric
def prediction_error(y, y_pred): 
    #return abs(y - y_pred)
    return 0




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
        plt.savefig(str(os.path.join( save_path , label+".png")), bbox_inches='tight')
    plt.show()







def run_training(model, optimizer, num_epochs, train_dataloader, val_dataloader, device, 
                 loss_fn=nn.MSELoss(), patience=1, early_stopper=None, scheduler=None, verbose=False):
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

    # initialize old loss value varibale
    val_loss_old = 0

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
                print("Early stopping")
                model = ES.load_checkpoint()
                break
        # Save biggest
        if early_stopper and epoch_val_loss > val_loss_old:
            val_loss_old = epoch_val_loss
            ES.save_model(model)
            
        if scheduler:
            scheduler.step(epoch_val_loss)

    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')

    plot("Loss", "Loss", train_losses, val_losses)

    return train_losses, val_losses



