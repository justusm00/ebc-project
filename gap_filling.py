import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


from modules.util import gap_filling_mlp
from columns import COLS_FEATURES, COLS_LABELS, COLS_TIME
from modules.MLPstuff import MLP


# SPECIFY THESE
normalization = True 
who_trained = 'JM' # author
num_hidden_units = 30
num_hidden_layers = 4
path_data = 'data/data_merged_with_nans.csv'



# construct model name
if normalization:
    model_name = f'mlp_nu{num_hidden_units}_nl{num_hidden_layers}_norm_{who_trained}'
else:
    model_name = f'mlp_nu{num_hidden_units}_nl{num_hidden_layers}_{who_trained}'

# construct model path
path_model = model_name + '.pth'

# Load the model
model = MLP(len(COLS_FEATURES), len(COLS_LABELS), num_hidden_units=30, num_hidden_layers=4)
model.load_state_dict(torch.load(path_model))



# load statistics
if normalization:
    # save statistics
    model_means_path = 'model_saves/' + model_name + '_means.npy'
    model_stds_path = 'model_saves/' + model_name + '_stds.npy'
    trainset_means = np.load(model_means_path)
    trainset_stds = np.load(model_stds_path)


# load data
data = pd.read_csv(path_data)
df_f = data.copy()

# Load the model
model = MLP(len(COLS_FEATURES), len(COLS_LABELS), num_hidden_units=30, num_hidden_layers=4)
model.load_state_dict(torch.load(path_model))

df_mlp = gap_filling_mlp(data=data, model=model, columns_data=COLS_FEATURES, columns_labels=COLS_LABELS, means=trainset_means, stds=trainset_stds)

# merge
df = df_mlp[COLS_TIME + ["H_f_mlp", "LE_f_mlp"]].merge(df_f, how='outer', on=COLS_TIME)

# filter by location and sort by date
df_bg = df[df['location'] == 0].sort_values(by=['year', 'month', 'day', '30min'])
df_gw = df[df['location'] == 1].sort_values(by=['year', 'month', 'day', '30min'])