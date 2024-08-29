###### This script is used for the actual filling of the gaps. Before running, make sure that you trained at least one RF and one MLP.
###### Optionally, you can specify another RF and MLP trained on different features / labels - taking into account the availability of the different features
###### All models must be trained on the same data - if artificial gaps are to be filled, none of the models must be trained on the artificial gaps
###### Optionally, you can specify if the test loss for each model should be printed




# important imports
import numpy as np
import pandas as pd
from modules.MLPstuff import compute_test_loss_rf, compute_test_loss_mlp
from modules.gapfilling_util import load_mlp, load_rf, gap_filling_mlp, gap_filling_rf
from modules.paths import PATH_PREPROCESSED, PATH_GAPFILLED




# SPECIFY THESE
filename_mlp = 'mlp_60_4_JM_minmax_AGF_5286a2e3c84ebdb055490bea6c9dc91c.pth' # mlp trained on important features
filename_mlpsw = 'mlp_60_4_JM_minmax_AGF_6ee83c392c0d7208dd385e8558700ff9.pth' # mlp trained on keys + incoming shortwave radiation
# filename_mlpsw = None
filename_rf = 'RF_AGF_5286a2e3c84ebdb055490bea6c9dc91c.pkl' # rf trained on important features
filename_rfsw = 'RF_AGF_6ee83c392c0d7208dd385e8558700ff9.pkl' # rf trained on keys + incoming shortwave radiation
# filename_rfsw = None


path_data = PATH_PREPROCESSED + 'data_merged_with_nans.csv'
print_test_loss = True





def fill_gaps(path_data,
              filename_mlp,
              filename_rf,
              filename_mlpsw=None,
              filename_rfsw=None,
              suffix_mlp='_f_mlp',
              suffix_rf='_f_rf',
              print_test_loss=True,
              diurnal_fill=None):
    """Perform gapfilling on data using pretrained mlp. Optionally, use MLP and Rf trained only on keys and incoming shortwave radiation to fill gaps where no other meteo data is available.

    Args:
        path_data (str): path to data (labeled and unlabeled)
        filename_mlp (str): name of file containing MLP parameters
        filename_rf (str): name of file containing RF parameters
        filename_mlpsw (str): name of file containing the MLP trained only on shortwave radiation and keys (optional)
        filename_rfsw (str): name of file containing the RF trained only on shortwave radiation and keys (optional)
        suffix_mlp (str): suffix added to mlp gapfilled columns (optional, defaults to '_f_mlp')
        suffix_rf (str): suffix added to rf gapfilled columns (optional, defaults to '_f_rf')
        print_test_loss (bool): whether or not test loss should be computed and printed (optional, defaults to True)

    """
    rf, hash_rf, cols_features_rf, cols_labels_rf, fill_artificial_gaps_rf = load_rf(filename_rf)


    if filename_rfsw:
        rfsw, hash_rfsw, cols_features_rfsw, cols_labels_rfsw, fill_artificial_gaps_rfsw = load_rf(filename_rfsw)
        if fill_artificial_gaps_rfsw != fill_artificial_gaps_rf:
            raise ValueError("RF and RFSW must be trained on the same data (one of them included the artificial gaps, one did not)")
        if (set(cols_labels_rfsw) != set(cols_labels_rf)):
            raise ValueError("Labels / target variables must be the same for both RFs")



    # load MLPs
    mlp, cols_features_mlp, cols_labels_mlp, hash_mlp, normalization_mlp, minmax_scaling_mlp, trainset_means,\
        trainset_stds, trainset_mins, trainset_maxs, fill_artificial_gaps_mlp  = load_mlp(filename_mlp)
    
    if fill_artificial_gaps_mlp != fill_artificial_gaps_rf:
        raise ValueError("MLP and RF must be trained on the same data (one of them included the artificial gaps, one did not)")


    if filename_mlpsw:
        mlpsw, cols_features_mlpsw, cols_labels_mlpsw, hash_mlpsw, normalization_mlpsw, minmax_scaling_mlpsw, \
            trainset_means_sw, trainset_stds_sw, trainset_mins_sw, trainset_maxs_sw, fill_artificial_gaps_mlpsw  = load_mlp(filename_mlpsw)
        if fill_artificial_gaps_mlpsw != fill_artificial_gaps_mlp:
            raise ValueError("MLP and MLPSW must be trained on the same data (one of them included the artificial gaps, one did not)")
        if (set(cols_labels_mlpsw) != set(cols_labels_mlp)):
            raise ValueError("Labels / target variables must be the same for both MLPs")



    if (set(cols_labels_rf) != set(cols_labels_mlp)):
        raise ValueError("Labels / target variables must be the same for MLP and RF")

        
    # now only use single variable for labels and fill_artificial_gaps
    cols_labels = cols_labels_rf
    fill_artificial_gaps = fill_artificial_gaps_rf

    if print_test_loss:
        # print RF test loss
        loss_test_rf = compute_test_loss_rf(rf,
                                            cols_features_rf,
                                            cols_labels_rf,
                                            hash_rf,
                                            fill_artificial_gaps)
        print(f"Test MSE for RF trained on {cols_features_rf}: {loss_test_rf:.2f}")
        # print MLP test loss
        loss_test_mlp = compute_test_loss_mlp(mlp,
                                        hash_mlp,
                                        cols_features_mlp,
                                        cols_labels_mlp,
                                        normalization_mlp,
                                        minmax_scaling_mlp,
                                        fill_artificial_gaps)
        print(f"Test MSE for MLP trained on {cols_features_mlp}: {loss_test_mlp:.2f}")
        if filename_rfsw:
            # print RFSW test loss
            loss_test_rfsw = compute_test_loss_rf(rfsw, cols_features_rfsw, cols_labels_rfsw, hash_rfsw, fill_artificial_gaps)
            print(f"Test MSE for RF trained on {cols_features_rfsw}: {loss_test_rfsw:.2f}")
        if filename_mlpsw:
            loss_test_mlpsw = compute_test_loss_mlp(mlpsw,
                                                hash_mlpsw,
                                                cols_features_mlpsw,
                                                cols_labels_mlpsw,
                                                normalization_mlpsw,
                                                minmax_scaling_mlpsw,
                                                fill_artificial_gaps)
            print(f"Test MSE for MLP trained on {cols_features_mlpsw}: {loss_test_mlpsw:.2f}")





      
    # load data
    data = pd.read_csv(path_data).drop(["H_f", "LE_f"], axis=1)

    if fill_artificial_gaps:
        # copy original values
        for col in cols_labels:
            data[col + '_copy'] = data[col]
        # set values in artificial gaps to na
        data.loc[data['artificial_gap'] != 0, cols_labels] = np.nan



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


    print("Total number of records:", data.shape[0])

    for col_mlp, col_rf in zip(cols_gapfilled_mlp, cols_gapfilled_rf):
        data[col_mlp] = df_mlp[col_mlp]
        data[col_rf] = df_rf[col_rf]
        print(f"Number of NaNs in {col_mlp}: {data[data[col_mlp].isna()].shape[0]}")
        print(f"Number of NaNs in {col_rf}: {data[data[col_rf].isna()].shape[0]}")


        if filename_mlpsw:
            data[col_mlp] = data[col_mlp].fillna(df_mlpsw[col_mlp])
            print(f"Number of NaNs in {col_mlp} after adding SW data: {data[data[col_mlp].isna()].shape[0]}")

        if filename_rfsw:
            data[col_rf] = data[col_rf].fillna(df_rfsw[col_rf])
            print(f"Number of NaNs in {col_rf} after adding SW data: {data[data[col_rf].isna()].shape[0]}")

    if fill_artificial_gaps:
        # retrieve original values
        for col in cols_labels:
            data = data.drop(col, axis=1)
            data[col] = data[col + '_copy']
            data = data.drop(col + '_copy', axis=1)



    # filter by location and sort by timestamps
    df_bg = data[data['location'] == 0].sort_values(by=['year', 'month', 'day', '30min'])
    df_gw = data[data['location'] == 1].sort_values(by=['year', 'month', 'day', '30min'])

    # drop location columns
    df_bg = df_bg.drop('location', axis=1)
    df_gw = df_gw.drop('location', axis=1)


    # save files
    df_bg.to_csv(PATH_GAPFILLED + 'BG_gapfilled.csv', index=False)
    df_gw.to_csv(PATH_GAPFILLED + 'GW_gapfilled.csv', index=False)



if __name__ == '__main__':
    fill_gaps(path_data=path_data, filename_mlp=filename_mlp, filename_rf=filename_rf, filename_mlpsw=filename_mlpsw,
              filename_rfsw=filename_rfsw, print_test_loss=print_test_loss)