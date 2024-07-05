# important  imports
import os
import numpy as np

import torch
import torch.nn as nn   
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



############# UTILITIES ############

from modules.util import EBCDataset, grab_data, train_val_splitter, data_loaders
from modules.MLPstuff import run_training, MLP, test
from columns import COLS_FEATURES, COLS_LABELS
from paths import PATH_MLP_TRAINING, PATH_MODEL_SAVES, PATH_PLOTS



# SPECIFY THESE
normalization = True 
who_trained = 'JM' # author
GPU = False
num_epochs = 100
lr = 10**(-3)
num_hidden_units = 60
num_hidden_layers = 4
batch_size = 10

# construct model name
if normalization:
    model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_norm_{who_trained}'
else:
    model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_{who_trained}'



def train_mlp(model_name, normalization, GPU, num_epochs, lr, path_model_saves, batch_size, path_plots):
    """Train MLP.

    Args:
        model_name (_type_): model name
        normalization (bool): If True, normalize trainset and save statistics.
        GPU (bool): if GPU should be used
        num_epochs (_type_): Number of epochs trained
        lr (_type_): learning rate
        path_model_saves (_type_): path where model and trainset statistics are saved
        batch_size (_type_):
        path_plots (str): path where plots are stored (as png)

    Returns:
        _type_: _description_
    """
    # Get number of cpus to use for faster parallelized data loading
    avb_cpus = os.cpu_count()
    num_cpus = 4
    print(avb_cpus, 'CPUs available,', num_cpus, 'were assigned' )

    # Device loader from Deep Learning

    ######## SPECIFY IF YOU DONT WANT TO USE CUDA (GPU) ###########
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


    trainset, testset = grab_data(PATH_MLP_TRAINING + 'training_data.csv', PATH_MLP_TRAINING + 'test_data.csv', num_cpus,
                                                            COLS_FEATURES, COLS_LABELS, normalization=normalization)

    trainset, valset = train_val_splitter(trainset)

    trainloader, valloader, testloader = data_loaders(trainset, valset, testset, num_cpus=num_cpus, batch_size=batch_size)


    if normalization:
        # save statistics
        model_means_path = 'model_saves/' + model_name + '_means.npy'
        model_stds_path = 'model_saves/' + model_name + '_stds.npy'
        np.save(model_means_path, trainset.dataset.means.numpy())
        np.save(model_stds_path, trainset.dataset.stds.numpy())
        print(f"Saved means to {model_means_path} \n")
        print(f"Saved stds to {model_stds_path} \n")




    print("Test run with small learning rate for single epoch ... \n")

    model = MLP(len(COLS_FEATURES), len(COLS_LABELS), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers).to(device)
    print("Model architecture: \n")
    print(model.eval())
    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)


    train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=1, train_dataloader=trainloader, val_dataloader=valloader, 
                                                                device=device, loss_fn=criterion, patience=5, early_stopper=False, verbose=False, plot_results=False)

    print(f"Initial train loss: {train_losses} \n")
    print(f"Initial val loss: {val_losses} \n")




    print("Beginning actual training ... \n")


    if GPU==True:
        torch.cuda.empty_cache()

    # Initialize the model
    model = MLP(len(COLS_FEATURES), len(COLS_LABELS), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers).to(device)
    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8,
                                threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5)



    train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=num_epochs, train_dataloader=trainloader, val_dataloader=valloader, 
                                                                device=device, loss_fn=criterion, patience=10, early_stopper=True, scheduler=scheduler, verbose=True, plot_results=True, save_plots_path=path_plots + 'loss/' + model_name + '_loss' + '.png')


    # Save the model
    model_save_path = path_model_saves + model_name + '.pth'
    torch.save(model.state_dict(), model_save_path )
    print(f"Saved model to {model_save_path} \n")


    test_loss = test(dataloader=testloader, model=model, device=device)
    print(f"Loss on testset: {test_loss} \n")



if __name__ == '__main__':
    train_mlp(model_name=model_name, normalization=normalization, GPU=GPU, num_epochs=num_epochs, lr=lr,
              path_model_saves=PATH_MODEL_SAVES, batch_size=batch_size, path_plots=PATH_PLOTS)
