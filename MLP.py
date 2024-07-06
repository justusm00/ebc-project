# important  imports
import os
import numpy as np
import hashlib
import json


import torch
import torch.nn as nn   
import torch.optim as optim



############# UTILITIES ############

from modules.util import grab_data, train_val_splitter, data_loaders, get_hash_from_features_and_labels
from modules.MLPstuff import run_training, MLP, test, MyReduceLROnPlateau
from columns import COLS_FEATURES_ALL, COLS_LABELS_ALL, COLS_KEY, COLS_KEY_ALT
from paths import PATH_MODEL_TRAINING, PATH_MODEL_SAVES_MLP, PATH_PLOTS



# SPECIFY THESE
cols_key = COLS_KEY # must be COLS_KEY or COLS_KEY_ALT
cols_features = cols_key + ["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit", "windSpeed"]
cols_labels = COLS_LABELS_ALL
normalization = False
minmax_scaling = True
who_trained = 'JM' # author
GPU = False
num_epochs = 150
lr = 10**(-3)
patience_early_stopper = 100
patience_scheduler = 10
num_hidden_units = 60
num_hidden_layers = 4
batch_size = 10




def train_mlp(GPU, num_epochs, lr,
              path_model_saves, batch_size, path_plots, cols_key, cols_features=COLS_FEATURES_ALL, cols_labels=COLS_LABELS_ALL, normalization=True, minmax_scaling=False, patience_early_stopper=10, patience_scheduler=10):
    """Train MLP.

    Args:
        GPU (bool): if GPU should be used
        num_epochs (_type_): Number of epochs trained
        lr (_type_): learning rate
        path_model_saves (_type_): path where model and trainset statistics are saved
        batch_size (_type_):
        path_plots (str): path where plots are stored (as png)
        cols_features (list): columns used as training features
        cols_labels (list): columns used as labels
        normalization (bool): If True, normalize trainset and save statistics.


    Returns:
        _type_: _description_
    """
    if (minmax_scaling is True ) and (normalization is True ) :
        raise ValueError("Can only perform normalization OR minmax_scaling")
    # check if time columns are present as features
    for col in cols_key:
        if col not in cols_features:
            raise ValueError(f"Features must contain all of {cols_key}")
        
    # Create a hash based on the features and labels
    model_hash = get_hash_from_features_and_labels(cols_features=cols_features, cols_labels=cols_labels)

    # construct model name
    if normalization:
        model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_{who_trained}_norm_{model_hash}'
    elif minmax_scaling:
        model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_{who_trained}_minmax_{model_hash}'
    else:
        model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_{who_trained}_{model_hash}'

    print("\n")
    print(f"Features used: {len(cols_features)} ({cols_features}) \n")
    print(f"Labels used: {len(cols_labels)} ({cols_labels}) \n")

    

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


    trainset, testset = grab_data(PATH_MODEL_TRAINING + 'training_data.csv', PATH_MODEL_TRAINING + 'test_data.csv',
                                  num_cpus, cols_features, cols_labels, normalization=normalization, minmax_scaling=minmax_scaling)

    trainset, valset = train_val_splitter(trainset, val_frac=0.2)

    trainloader, valloader, testloader = data_loaders(trainset, valset, testset,
                                                      num_cpus=num_cpus, batch_size=batch_size)



    # print("Test run with small learning rate for single epoch ... \n")

    # model = MLP(len(cols_features), len(cols_labels), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers).to(device)
    # print("Model architecture: \n")
    # print(model.eval())
    # # Set loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)


    # train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=1,
    #                                          train_dataloader=trainloader, val_dataloader=valloader,
    #                                          device=device, loss_fn=criterion, patience=5, early_stopper=False, verbose=False, plot_results=False)

    # print(f"Initial train loss: {train_losses} \n")
    # print(f"Initial val loss: {val_losses} \n")




    print("Beginning actual training ... \n")


    if GPU==True:
        torch.cuda.empty_cache()

    # Initialize the model
    model = MLP(len(cols_features), len(cols_labels), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers).to(device)
    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MyReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_scheduler,
                                threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5)



    train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=num_epochs, 
                                            train_dataloader=trainloader, val_dataloader=valloader, device=device, loss_fn=criterion, patience=patience_early_stopper, 
                                            early_stopper=True, scheduler=scheduler, verbose=True, plot_results=True,
                                            save_plots_path=path_plots + 'loss/' + model_name + '.png')


    # Save the model
    model_save_path = path_model_saves + model_name + '.pth'
    torch.save(model.state_dict(), model_save_path )
    print(f"Saved model to {model_save_path} \n")

    # save features and labels
    features_json = path_model_saves + 'features/' + model_name + '.json'
    labels_json = path_model_saves + 'labels/' + model_name + '.json'
    with open(features_json, 'w') as file:
        json.dump(cols_features, file)
    with open(labels_json, 'w') as file:
        json.dump(cols_labels, file)

    print(f"Saved list of features to {features_json} \n")

    print(f"Saved list of labels to {labels_json} \n")


    if normalization:
        # save statistics
        model_means_path = path_model_saves + 'statistics/' + model_name + '_means.npy'
        model_stds_path = path_model_saves + 'statistics/'  + model_name + '_stds.npy'
        np.save(model_means_path, trainset.dataset.means.numpy())
        np.save(model_stds_path, trainset.dataset.stds.numpy())
        print(f"Saved means to {model_means_path} \n")
        print(f"Saved stds to {model_stds_path} \n")

    if minmax_scaling:
        # save statistics
        model_maxs_path = path_model_saves + 'statistics/' + model_name + '_maxs.npy'
        model_mins_path = path_model_saves + 'statistics/'  + model_name + '_mins.npy'
        np.save(model_maxs_path, trainset.dataset.maxs.numpy())
        np.save(model_mins_path, trainset.dataset.mins.numpy())
        print(f"Saved maxs to {model_maxs_path} \n")
        print(f"Saved mins to {model_mins_path} \n")


    test_loss = test(dataloader=testloader, model=model, device=device)
    print(f"Loss on testset: {test_loss} \n")



if __name__ == '__main__':
    train_mlp(GPU=GPU, num_epochs=num_epochs, lr=lr,
              path_model_saves=PATH_MODEL_SAVES_MLP, batch_size=batch_size,
              path_plots=PATH_PLOTS, cols_key=cols_key, cols_features=cols_features,
              cols_labels=cols_labels, normalization=normalization,
              minmax_scaling=minmax_scaling, patience_early_stopper=patience_early_stopper,
              patience_scheduler=patience_scheduler)
