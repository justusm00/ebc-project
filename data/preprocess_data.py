import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

import sys
sys.path.append('../ebc-project')
from modules.util import transform_timestamp, numerical_to_float

PATH = 'data/data_files/'
# numerical values to transform to float
COLS_NUMERICAL_PREP = ['H_orig', 'LE_orig', 'ET_orig', 'CO2', 'H2O', 'NEE_orig', 'Reco', 'GPP_f', 'Ustar']
COLS_NUMERICAL_ORIG = ['H_orig', 'LE_orig', 'ET_orig', 'CO2', 'H2O', 'NEE_orig', 'Reco', 'GPP_f', 'Ustar', 'H_f', 'LE_f', 'ET_f', 'NEE_f']
# unnecessary columns to be dropped
COLS_DROP_PREP = ['TIMESTAMP_START', 'TIMESTAMP_MITTE', 'TIMESTAMP_ENDE', 'H_f', 'LE_f', 'ET_f', 'NEE_f']
COLS_DROP_ORIG = ['TIMESTAMP_START', 'TIMESTAMP_MITTE', 'TIMESTAMP_ENDE']

# collect files to preprocess
files = [f for f in os.listdir(PATH) if '.csv' in f]

data = []
count_na = []

for f in tqdm(files):
    """ For united dataset """
    try: 
        df = pd.read_csv(PATH+f, sep=',').drop(0)
    except pd.errors.ParserError: 
        # the 2024 files use ';' as separator and ',' as decimal separator
        df = pd.read_csv(PATH+f, sep=';').drop(0)

    # location based on file name (files should be properly labelled with either BG or GW!)
    # one-hot encode the location: BG (botanical garden)==0, GW (Goettinger forest)==1
    df['location'] = '0' if 'BG' in f else '1'

    df = transform_timestamp(df, 'TIMESTAMP_START')
    df = numerical_to_float(df, COLS_NUMERICAL_PREP)
    df.drop(COLS_DROP_PREP, axis=1, inplace=True)

    # drop any row containing NA values
    len_before = df.__len__()
    df.dropna(axis=0, how='any', inplace=True, ignore_index=True)
    na_removed = len_before - df.__len__()
    count_na.append(na_removed)

    data.append(df)

    """ For preparing the original data for fitting and method comparison """
    try: 
        df = pd.read_csv(PATH+f, sep=',', na_values=['NaN']).drop(0)
    except pd.errors.ParserError: 
        # the 2024 files use ';' as separator and ',' as decimal separator
        df = pd.read_csv(PATH+f, sep=';', na_values=['NaN']).drop(0)

    df['location'] = '0' if 'BG' in f else '1'

    df = transform_timestamp(df, 'TIMESTAMP_START')
    df = numerical_to_float(df, COLS_NUMERICAL_ORIG)
    df.drop(COLS_DROP_ORIG, axis=1, inplace=True)

    # rename the files to name_mod.csv
    name, csv = os.path.splitext(f)
    mod_name = name + '_mod' + csv

    df.to_csv(os.path.join("data/data_files_modified/", mod_name))



    print(f'{f} done')

print(f'\nRows removed because of NA: {sum(count_na)}\n')

# combine the preprocessed data into single dataframe
data_final = pd.concat(data, axis=0, ignore_index=True)

# rescaling all numerical values to be in range [0,1]
# alternatively center around 0 with unit std?
for col, type in zip(data_final.columns, data_final.dtypes):
    if type == 'float64':
        data_final[f'{col}'] = minmax_scale(data_final[f'{col}'])

data_final.to_csv('data/data_preprocessed.csv')