import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


from modules.util import gap_filling_mlp
from columns import COLS_FEATURES, COLS_LABELS, COLS_TIME
from paths import PATH_PREPROCESSED, PATH_GAPFILLED, PATH_MODEL_SAVES
from modules.MLPstuff import MLP


# SPECIFY THESE
normalization = True 
who_trained = 'JM' # author
num_hidden_units = 30
num_hidden_layers = 4
path_data = PATH_PREPROCESSED + 'data_merged_with_nans.csv'



# construct model name
if normalization:
    model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_norm_{who_trained}'
else:
    model_name = f'mlp_{num_hidden_units}_{num_hidden_layers}_{who_trained}'


def fill_gaps(model_name, path_data, path_gapfilled, path_model_saves):
    # construct model path
    path_model = path_model_saves + model_name + '.pth'

    # Load the model
    model = MLP(len(COLS_FEATURES), len(COLS_LABELS), num_hidden_units=30, num_hidden_layers=4)
    model.load_state_dict(torch.load(path_model))



    # load statistics
    if normalization:
        # save statistics
        model_means_path = path_model_saves + model_name + '_means.npy'
        model_stds_path = path_model_saves + model_name + '_stds.npy'
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

    # reconstruct timestamp column

    # Convert the '30min' column to a timedelta representing the minutes
    df['time'] = pd.to_timedelta(df['30min'] * 30, unit='m')

    # Create the datetime column by combining 'year', 'month', 'day' and 'time'
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']]) + df['time']

    # drop year, month, day columns
    df = df.drop(['year', 'month', 'day', '30min', 'time', 'day_of_year'], axis=1)


    # filter by location and sort by timestamps
    df_bg = df[df['location'] == 0].sort_values(by='timestamp')
    df_gw = df[df['location'] == 1].sort_values(by='timestamp')

    # drop location columns
    df_bg = df_bg.drop('location', axis=1)
    df_gw = df_gw.drop('location', axis=1)


    # save files
    df_bg.to_csv(path_gapfilled + 'BG_gapfilled.csv', index=False)
    df_gw.to_csv(path_gapfilled + 'GW_gapfilled.csv', index=False)



if __name__ == '__main__':
    fill_gaps(model_name=model_name, path_data=path_data, path_gapfilled=PATH_GAPFILLED, path_model_saves=PATH_MODEL_SAVES)