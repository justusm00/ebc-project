import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from modules.util import gap_filling_mlp, gap_filling_rf
from modules.MLPstuff import compute_test_loss_rf, compute_test_loss_mlp
from modules.gapfilling_util import load_mlp, load_rf
from modules.paths import PATH_PREPROCESSED, PATH_GAPFILLED


# SPECIFY THESE
filename_mlp = 'mlp_60_4_JM_minmax_01b3187c62d1a0ed0d00b5736092b0d1.pth' # mlp trained on important features
filename_mlpsw = 'mlp_60_4_JM_minmax_d2b43b2dba972e863e8a9a0deeaebbda.pth' # mlp trained on keys + incoming shortwave radiation
# filename_mlpsw = None
filename_rf = 'RandomForest_model_dea73fac9940fa6b1ad3defbe517d876.pkl' # rf trained on important features
filename_rfsw = 'RandomForest_model_3311a46a744485acf24fe5c02b8f8dab.pkl' # rf trained on keys + incoming shortwave radiation
# filename_rfsw = None

path_data = PATH_PREPROCESSED + 'data_merged_with_nans.csv'



def fill_gaps(path_data,
              filename_mlp,
              filename_rf,
              filename_mlpsw=None,
              filename_rfsw=None,
              suffix_mlp='_f_mlp',
              suffix_rf='_f_rf',
              diurnal_fill=None):
    """Perform gapfilling on data using pretrained mlp. Optionally, use MLP and Rf trained only on keys and incoming shortwave radiation to fill gaps where no other meteo data is available.

    Args:
        path_data (str): path to data (labeled and unlabeled)
        filename_mlp (str): name of file containing MLP parameters
        filename_rf (str): name of file containing RF parameters
        filename_mlpsw (str): name of file containing the MLP trained only on shortwave radiation and keys (optional)
        filename_rfsw (str): name of file containing the RF trained only on shortwave radiation and keys (optional)

    """
    rf, hash_rf, cols_features_rf, cols_labels_rf = load_rf(filename_rf)


    # print RF test loss
    loss_test_rf = compute_test_loss_rf(rf, cols_features_rf, cols_labels_rf, hash_rf)
    print(f"Test MSE for RF trained on {cols_features_rf}: {loss_test_rf:.2f}")

    if filename_rfsw:
        rfsw, hash_rfsw, cols_features_rfsw, cols_labels_rfsw = load_rf(filename_rfsw)


        # print RF test loss
        loss_test_rfsw = compute_test_loss_rf(rfsw, cols_features_rfsw, cols_labels_rfsw, hash_rfsw)
        print(f"Test MSE for RF trained on {cols_features_rfsw}: {loss_test_rfsw:.2f}")





    # load MLPs
    mlp, cols_features_mlp, cols_labels_mlp, hash_mlp, normalization_mlp, minmax_scaling_mlp, trainset_means,\
        trainset_stds, trainset_mins, trainset_maxs  = load_mlp(filename_mlp)
    loss_test_mlp = compute_test_loss_mlp(mlp,
                                          hash_mlp,
                                          cols_features_mlp,
                                          cols_labels_mlp,
                                          normalization_mlp,
                                          minmax_scaling_mlp)
    print(f"Test MSE for MLP trained on {cols_features_mlp}: {loss_test_mlp:.2f}")

    if filename_mlpsw:
        mlpsw, cols_features_mlpsw, cols_labels_mlpsw, hash_mlpsw, normalization_mlpsw, minmax_scaling_mlpsw, \
            trainset_means_sw, trainset_stds_sw, trainset_mins_sw, trainset_maxs_sw  = load_mlp(filename_mlpsw)
        loss_test_mlpsw = compute_test_loss_mlp(mlpsw,
                                                hash_mlpsw,
                                                cols_features_mlpsw,
                                                cols_labels_mlpsw,
                                                normalization_mlpsw,
                                                minmax_scaling_mlpsw)
        print(f"Test MSE for MLP trained on {cols_features_mlpsw}: {loss_test_mlpsw:.2f}")



    if (set(cols_labels_rf) != set(cols_labels_mlp)):
        raise ValueError("Labels / target variables must be the same for all models")
    if filename_mlpsw:
        if (set(cols_labels_mlpsw) != set(cols_labels_mlp)):
            raise ValueError("Labels / target variables must be the same for all models")
        
    # now only use single variable for labels
    cols_labels = cols_labels_rf

      
    # load data
    data = pd.read_csv(path_data)


    # get gapfilled dataframes
    df_mlp = gap_filling_mlp(data=data, mlp=mlp, cols_features=cols_features_mlp,
                             cols_labels=cols_labels, suffix=suffix_mlp, means=trainset_means, stds=trainset_stds,
                             mins=trainset_mins, maxs=trainset_maxs)

    df_rf = gap_filling_rf(data=data, model=rf, cols_features=cols_features_rf,
                           cols_labels=cols_labels, suffix=suffix_rf)


    if filename_mlpsw:
        df_mlpsw = gap_filling_mlp(data=data, mlp=mlpsw,
                                   cols_features=cols_features_mlpsw,
                                   cols_labels=cols_labels, suffix=suffix_mlp, means=trainset_means_sw, stds=trainset_stds_sw,
                                   mins=trainset_mins_sw, maxs=trainset_maxs_sw)
        
    if filename_rfsw:
        df_rfsw = gap_filling_rf(data=data, model=rfsw,
                                 cols_features=cols_features_rfsw, cols_labels=cols_labels, suffix=suffix_rf)

    


    cols_gapfilled_mlp = [col.replace('_orig', '') + suffix_mlp for col in cols_labels]
    cols_gapfilled_rf = [col.replace('_orig', '') + suffix_rf for col in cols_labels]
    cols_gapfilled_mds = [col.replace('_orig', '') + '_f' for col in cols_labels]


    print("Total number of records:", data.shape[0])

    for col_mlp, col_rf in zip(cols_gapfilled_mlp, cols_gapfilled_rf):
        data[col_mlp] = df_mlp[col_mlp]
        data[col_rf] = df_rf[col_rf]
        print(f"Number of NaNs in {col_mlp}: {data[data[col_mlp].isna()].shape[0]}")
        print(f"Number of NaNs in {col_rf}: {data[data[col_rf].isna()].shape[0]}")

        if filename_mlpsw:
            data[col_mlp] = data[col_mlp].fillna(df_mlpsw[col_mlp])
            print(f"Number of NaNs in {col_mlp} after adding SW data: {data[data[col_mlp].isna()].shape[0]}")

        if filename_rf:
            data[col_rf] = data[col_rf].fillna(df_rfsw[col_rf])
            print(f"Number of NaNs in {col_rf} after adding SW data: {data[data[col_rf].isna()].shape[0]}")




    # Convert the '30min' column to a timedelta representing the minutes
    data['time'] = pd.to_timedelta(data['30min'] * 30, unit='m')

    # Create the datetime column by combining 'year', 'month', 'day' and 'time'
    data['timestamp'] = pd.to_datetime(data[['year', 'month', 'day']]) + data['time']

    # drop year, month, day columns
    data = data.drop(['year', 'month', 'day', '30min', 'time'], axis=1)


    # filter by location and sort by timestamps
    df_bg = data[data['location'] == 0].sort_values(by='timestamp')
    df_gw = data[data['location'] == 1].sort_values(by='timestamp')

    # drop location columns
    df_bg = df_bg.drop('location', axis=1)
    df_gw = df_gw.drop('location', axis=1)


    # save files
    df_bg.to_csv(PATH_GAPFILLED + 'BG_gapfilled.csv', index=False)
    df_gw.to_csv(PATH_GAPFILLED + 'GW_gapfilled.csv', index=False)



if __name__ == '__main__':
    fill_gaps(path_data=path_data, filename_mlp=filename_mlp, filename_rf=filename_rf, filename_mlpsw=filename_mlpsw, filename_rfsw=filename_rfsw)