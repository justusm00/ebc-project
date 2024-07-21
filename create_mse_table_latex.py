import pandas as pd
import numpy as np
from modules.paths import PATH_GAPFILLED


###### Script to compute MSES for RF, MLP and MDS. Computes per site, artificial gap size and variable.
###### Need to run gap_filling.py and gap_filling_mds.R first
###### Currently only works for variables H and LE
###### Resulting df is saved directly as latex table




def create_latex_table():
    # load RF and MLP data
    df_bg = pd.read_csv(PATH_GAPFILLED + 'BG_gapfilled.csv')
    df_gw = pd.read_csv(PATH_GAPFILLED + 'GW_gapfilled.csv')
    df_bg["location"] = 0
    df_gw["location"] = 1
    df = pd.concat([df_bg, df_gw]).reset_index(drop=True)



    # load and merge MDS data
    df_mds = pd.read_csv(PATH_GAPFILLED + 'combined_gapfilled_mds.csv')
    df_mds["30min"] = df_mds["X30min"]
    df_mds.drop("X30min", axis=1, inplace=True)
    df_mds["H_f_mds"] = df_mds["H_f"]
    df_mds["LE_f_mds"] = df_mds["LE_f"]
    df_mds.drop(["H_f", "LE_f"], axis=1, inplace=True)

    df = df.merge(df_mds, on=["year", "day_of_year", "30min", "location"], how="outer")



    def get_mse(df, model_suffix, locs, variables, gap_sizes):
        n_variables = len(variables)
        variables_pred = [col.replace('_orig', '') + model_suffix  for col in variables]
        df = df[(df["artificial_gap"].isin(gap_sizes)) & (df["location"].isin(locs))].loc[df[variables + variables_pred].notna().all(axis=1)]
        mse = 0
        for var, var_pred in zip(variables, variables_pred):
            mse = mse + np.mean((df[var] - df[var_pred])**2)
        mse = mse / n_variables
        return mse

    locs_all = [[0], [1], [0, 1]]
    locs_all_names = ['BG', 'GW', 'BG, GW']
    gap_sizes_all = [[1], [2], [3], [1, 2, 3]]
    gap_sizes_all_names = ["1", "7", "21", "1, 7, 21"]
    variables=["H_orig", "LE_orig"]

    df_mses = pd.DataFrame(columns=["Site", "Gap-size [days]", "MSE(RF)", "MSE(MLP)", "MSE(MDS)"])
    i = 0
    for locs, locs_names in zip(locs_all, locs_all_names):
        for gap_sizes, gap_sizes_names in zip(gap_sizes_all, gap_sizes_all_names):
            i = i + 1
            mse_rf = get_mse(df, '_f_rf', locs, variables, gap_sizes)
            mse_mlp = get_mse(df, '_f_mlp', locs, variables, gap_sizes)
            mse_mds = get_mse(df, '_f_mds', locs, variables, gap_sizes)
            new_row = {'Site': locs_names, 'Gap-size [days]': gap_sizes_names, 'MSE(RF)': mse_rf, 'MSE(MLP)': mse_mlp, 'MSE(MDS)': mse_mds}
            df_mses = pd.concat([df_mses, pd.DataFrame(new_row, index=[i])])

    df_mses["MSE(RF)"] = df_mses["MSE(RF)"].round(2)
    df_mses["MSE(MLP)"] = df_mses["MSE(MLP)"].round(2)

    with open('tables/mse_latex.tex', 'w') as f:
        f.write(df_mses.to_latex(index=False,
                                float_format='{:.2f}'.format,
                                column_format='|c|c|c|c|c|c|c|'
                                    
    ))
        
    return 
        

if __name__ == '__main__':
    create_latex_table()