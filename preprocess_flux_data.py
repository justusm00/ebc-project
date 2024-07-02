import os
import pandas as pd
from tqdm import tqdm

from modules.util import transform_timestamp, numerical_to_float

PATH = 'data/data_files/'
# numerical values to transform to float
COLS_NUMERICAL = ['H_orig', 'LE_orig', 'ET_orig', 'CO2', 'H2O', 'NEE_orig', 'GPP_f', 'Ustar', 'H_f', 'LE_f', 'ET_f', 'NEE_f']

# unnecessary columns to be dropped
# remove Reco because it is already gapfilled
COLS_DROP_ORIG = ['TIMESTAMP_MITTE', 'TIMESTAMP_ENDE', 'Reco']
# additional columns to be dropped for training data
COLS_DROP_PREP = ['H_f', 'LE_f', 'ET_f', 'NEE_f', 'GPP_f']

# collect files to preprocess
files = [f for f in os.listdir(PATH) if 'fluxes' in f]


data_prep = []
data_orig = []
count_na = []

for f in tqdm(files):
    try: 
        df = pd.read_csv(PATH+f, sep=',').drop(0)
    except pd.errors.ParserError: 
        # the 2024 files use ';' as separator and ',' as decimal separator
        df = pd.read_csv(PATH+f, sep=';').drop(0)

    # location based on file name (files should be properly labelled with either BG or GW!)
    # one-hot encode the location: BG (botanical garden)==0, GW (Goettinger forest)==1
    df['location'] = '0' if 'BG' in f else '1'

    df = transform_timestamp(df, 'TIMESTAMP_START')
    df = numerical_to_float(df, COLS_NUMERICAL)


    df.drop(COLS_DROP_ORIG, axis=1, inplace=True)
    data_orig.append(df.copy())

    # drop further columns
    df.drop(COLS_DROP_PREP, axis=1, inplace=True)

    # drop any row containing NA values
    len_before = df.__len__()
    df.dropna(axis=0, how='any', inplace=True, ignore_index=True)
    na_removed = len_before - df.__len__()
    count_na.append(na_removed)

    data_prep.append(df.copy())


    print(f'{f} done')

print(f'\nRows removed because of NA: {sum(count_na)}\n')

# combine the preprocessed data into single dataframe
data_prep_final = pd.concat(data_prep, axis=0, ignore_index=True)
data_orig_final = pd.concat(data_orig, axis=0, ignore_index=True)




data_prep_final.to_csv('data/flux_data_preprocessed.csv', index=False)
data_orig_final.to_csv('data/flux_data_preprocessed_gapfilled.csv', index=False)