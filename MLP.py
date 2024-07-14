# important  imports
import os
import numpy as np
import json


import torch
import torch.nn as nn   
import torch.optim as optim



############# UTILITIES ############

from modules.util import get_hash_from_features_and_labels
from modules.dataset_util import grab_data, train_val_splitter, data_loaders,train_test_splitter, SingleBatchDataLoader
from modules.MLPstuff import run_training, MLP, test, MyReduceLROnPlateau
from modules.columns import COLS_FEATURES_ALL, COLS_LABELS_ALL, COLS_KEY, COLS_KEY_ALT
from modules.paths import PATH_MODEL_TRAINING, PATH_MODEL_SAVES_MLP, PATH_PLOTS, PATH_PREPROCESSED,\
    PATH_MODEL_SAVES_FEATURES, PATH_MODEL_SAVES_LABELS, PATH_MODEL_SAVES_STATISTICS



# SPECIFY THESE
cols_features = ["incomingShortwaveRadiation", "location", "soilTemperature",
                 "windSpeed", "airTemperature", "30min", "day_of_year"]
# cols_features = COLS_KEY + ["incomingShortwaveRadiation"]
cols_labels = COLS_LABELS_ALL
normalization = False
minmax_scaling = True
who_trained = 'JM' # author
GPU = False
num_epochs = 150
lr = 10**(-3)
patience_early_stopper = 20
patience_scheduler = 10
num_hidden_units = 60
num_hidden_layers = 8
batch_size = 20




def train_mlp(GPU, num_epochs, lr, batch_size, cols_features=COLS_FEATURES_ALL, 
              cols_labels=COLS_LABELS_ALL,
              normalization=True, minmax_scaling=False, 
              patience_early_stopper=10, patience_scheduler=10):
    """Train MLP.

    Args:
        GPU (bool): if GPU should be used
        num_epochs (_type_): Number of epochs trained
        lr (_type_): learning rate
        batch_size (_type_):
        cols_features (list): columns used as training features
        cols_labels (list): columns used as labels
        normalization (bool): If True, normalize trainset and save statistics.


    Returns:
        _type_: _description_
    """
    # sort features and labels
    cols_features = sorted(cols_features)
    cols_labels = sorted(cols_labels)
    if (minmax_scaling is True ) and (normalization is True ) :
        raise ValueError("Can only perform normalization OR minmax_scaling")        
        
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

    if GPU==True:
        torch.cuda.empty_cache()

    # try to load train and test data. If not available, create train / test split for this combination of features and labels
    try:
        trainset, testset = grab_data(path_train=PATH_MODEL_TRAINING + 'training_data_' + model_hash + '.csv', 
                                path_test=PATH_MODEL_TRAINING + 'test_data_' + model_hash + '.csv', num_cpus=num_cpus, cols_features=cols_features, cols_labels=cols_labels, normalization=normalization, minmax_scaling=minmax_scaling)
    except:
        print("No train and test data available for given feature/label combination. Creating one ... \n")
        train_test_splitter(path_data=PATH_PREPROCESSED + 'data_merged_with_nans.csv', 
                               cols_features=cols_features, 
                               cols_labels=cols_labels, 
                               path_save=PATH_MODEL_TRAINING, model_hash=model_hash)
        trainset, testset = grab_data(path_train=PATH_MODEL_TRAINING + 'training_data_' + model_hash + '.csv', 
                                path_test=PATH_MODEL_TRAINING + 'test_data_' + model_hash + '.csv', num_cpus=num_cpus, cols_features=cols_features, cols_labels=cols_labels, normalization=normalization, minmax_scaling=minmax_scaling)


    trainset, valset = train_val_splitter(trainset, val_frac=0.2)

    trainloader, valloader, testloader = data_loaders(trainset, valset, testset,
                                                      num_cpus=num_cpus, batch_size=batch_size)
    

    # create single batch dataloaders for intentional overfitting

    trainloader_of = SingleBatchDataLoader(trainloader)
    valloader_of = SingleBatchDataLoader(valloader)



    print("Trying to overfit on single batch ... \n")

    model = MLP(len(cols_features), len(cols_labels), num_hidden_units=num_hidden_units, num_hidden_layers=num_hidden_layers).to(device)
    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    train_losses, val_losses = run_training(model=model, optimizer=optimizer, num_epochs=1000,
                                             train_dataloader=trainloader_of, val_dataloader=valloader_of,
                                             device=device, loss_fn=criterion, patience=5, early_stopper=False, verbose=False, plot_results=True)





    print("Beginning actual training ... \n")
    
    print("Model architecture: \n")
    print(model.eval())

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
                                            save_plots_path=PATH_PLOTS + 'loss/' + model_name + '.png')


    # Save the model
    model_save_path = PATH_MODEL_SAVES_MLP + model_name + '.pth'
    torch.save(model.state_dict(), model_save_path )
    print(f"Saved model to {model_save_path} \n")

    # save features and labels
    features_json = PATH_MODEL_SAVES_FEATURES + model_hash+ '.json'
    labels_json = PATH_MODEL_SAVES_LABELS + model_hash + '.json'
    with open(features_json, 'w') as file:
        json.dump(cols_features, file)
    with open(labels_json, 'w') as file:
        json.dump(cols_labels, file)

    print(f"Saved list of features to {features_json} \n")

    print(f"Saved list of labels to {labels_json} \n")


    if normalization:
        # save statistics
        model_means_path = PATH_MODEL_SAVES_STATISTICS + model_hash + '_means.npy'
        model_stds_path = PATH_MODEL_SAVES_STATISTICS  + model_hash + '_stds.npy'
        np.save(model_means_path, trainset.dataset.means.numpy())
        np.save(model_stds_path, trainset.dataset.stds.numpy())
        print(f"Saved means to {model_means_path} \n")
        print(f"Saved stds to {model_stds_path} \n")

    if minmax_scaling:
        # save statistics
        model_maxs_path = PATH_MODEL_SAVES_STATISTICS + model_hash + '_maxs.npy'
        model_mins_path = PATH_MODEL_SAVES_STATISTICS + model_hash  + '_mins.npy'
        np.save(model_maxs_path, trainset.dataset.maxs.numpy())
        np.save(model_mins_path, trainset.dataset.mins.numpy())
        print(f"Saved maxs to {model_maxs_path} \n")
        print(f"Saved mins to {model_mins_path} \n")


    test_loss = test(dataloader=testloader, model=model, device=device)
    print(f"Loss on testset: {test_loss:.2f} \n")



if __name__ == '__main__':
    train_mlp(GPU=GPU,
              num_epochs=num_epochs,
              lr=lr,
              batch_size=batch_size,
              cols_features=cols_features,
              cols_labels=cols_labels,
              normalization=normalization,
              minmax_scaling=minmax_scaling,
              patience_early_stopper=patience_early_stopper,
              patience_scheduler=patience_scheduler)
