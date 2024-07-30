import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.paths import PATH_GAPFILLED, PATH_PLOTS
plt.rcParams['font.size'] = 14



###### Script to compute MSES for RF, MLP and MDS. Computes per site, artificial gap size and variable. Creates barplots for each variable
###### Need to run gap_filling.py and gap_filling_mds.R first
###### Currently only works for variables H and LE
###### Resulting plots are saved 




def plot_rsr():
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


    def get_rsr(df, model_suffix, locs, variable, gap_sizes):
        variable_pred = variable.replace('_orig', '') + model_suffix
        # compute variances for location
        variance = df[(df["location"].isin(locs))].loc[df[[variable, variable_pred]].notna().all(axis=1)][variable].var()
        df = df[(df["artificial_gap"].isin(gap_sizes)) & (df["location"].isin(locs))].loc[df[[variable, variable_pred]].notna().all(axis=1)]
        rsr = np.sqrt(np.mean((df[variable] - df[variable_pred])**2) / variance)
        return rsr
    locs_all = [[0], [1], [0, 1]]
    locs_all_names = ['BG', 'GW', 'BG, GW']
    gap_sizes_all = [[1], [2], [3], [1, 2, 3]]
    gap_sizes_all_names = ["1", "7", "21", "1, 7, 21"]



    def plot_rsr_for_var(variable):
        fig, axs = plt.subplots(1, len(locs_all), figsize=(15, 5))
        
        if len(locs_all) == 1:
            axs = [axs]
        
        for ax, locs, locs_name in zip(axs, locs_all, locs_all_names):
            rsr_rf_list = []
            rsr_mlp_list = []
            rsr_mds_list = []
            gap_sizes_list = []

            for gap_sizes, gap_sizes_name in zip(gap_sizes_all, gap_sizes_all_names):
                rsr_rf = get_rsr(df, '_f_rf', locs, variable, gap_sizes)
                rsr_mlp = get_rsr(df, '_f_mlp', locs, variable, gap_sizes)
                rsr_mds = get_rsr(df, '_f_mds', locs, variable, gap_sizes)

                rsr_rf_list.append(rsr_rf)
                rsr_mlp_list.append(rsr_mlp)
                rsr_mds_list.append(rsr_mds)
                gap_sizes_list.append(gap_sizes_name)
            
            # Bar width
            bar_width = 0.2
            # Gap size indices for x-axis
            indices = np.arange(len(gap_sizes_list))
            
            ax.bar(indices - bar_width, rsr_rf_list, bar_width, label='RF', capsize=5)
            ax.bar(indices, rsr_mlp_list, bar_width, label='MLP', capsize=5)
            ax.bar(indices + bar_width, rsr_mds_list, bar_width, label='MDS', capsize=5)

            ax.set_xlabel('Gap Size')
            ax.set_ylabel('RSR')
            ax.set_title(f'RSR for {locs_name}')
            ax.set_xticks(indices)
            ax.set_xticklabels(gap_sizes_list)
            ax.legend()

        plt.tight_layout()
        plt.savefig(PATH_PLOTS + f'gapfilling_comparison/rsr_comparison_{variable}.pdf', dpi=150)


    plot_rsr_for_var("H_orig")
    plot_rsr_for_var("LE_orig")


         
        

if __name__ == '__main__':
    plot_rsr()