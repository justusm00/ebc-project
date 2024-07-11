import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.paths import PATH_GAPFILLED, PATH_PLOTS


def plot_diurnal_cycles(path_gapfilled, path_plots):
    """Load gapfilled data and plot diurnal cycles. Compare different gapfilling methods and different locations (BG and GW).

    Args:
        path_gapfilled (_type_): _description_
        path_plots (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load files
    df_gw = pd.read_csv(PATH_GAPFILLED + 'GW_gapfilled.csv')
    df_bg = pd.read_csv(PATH_GAPFILLED + 'BG_gapfilled.csv')

    # filter for 2024
    df_gw = df_gw[df_gw["timestamp"] > '2024-01-01 00:00:00']
    df_bg = df_bg[df_bg["timestamp"] > '2024-01-01 00:00:00']

    # add time column
    df_gw['time'] = pd.to_datetime(df_gw['timestamp']).dt.strftime('%H:%M:%S')
    df_bg['time'] = pd.to_datetime(df_bg['timestamp']).dt.strftime('%H:%M:%S')

    # add net radiation
    df_gw["Q"] = df_gw["netRadiation"]  
    df_bg["Q"] = df_bg["netRadiation"]


    def aggregate_data(df):
        df_agg = df.groupby("time").agg(G=("soilHeatflux", "mean"), H_orig = ("H_orig", "mean"), H_f=("H_f", "mean"),
                                        H_f_mlp=("H_f_mlp", "mean"), H_f_rf=("H_f_rf", "mean"),  LE_orig=("LE_orig", "mean"), LE_f=("LE_f", "mean"),
                                        LE_f_mlp=("LE_f_mlp", "mean"), LE_f_rf=("LE_f_rf", "mean"), Q=("Q", "mean")).reset_index()
        return df_agg


    df_gw = aggregate_data(df_gw)
    df_bg = aggregate_data(df_bg)

    # # plot diurnal cycles
    fig, axs = plt.subplots(4, 2, figsize =(20, 8), dpi=600)

    for ax in axs.flatten():
        ax.set_xticks([])  # Remove x-ticks
        ax.set_xticklabels([])

    fig.suptitle("Energy Balance Closure components")
    axs[0, 0].set_title("Botanical Garden")
    axs[0, 1].set_title("Göttingen Forest")
    for i, df, location in zip([0, 1], [df_bg, df_gw], ["BG", "GW"]):
        for j, col in enumerate(["Q", "H_orig", "LE_orig" ,"G"]):
            axs[j, i].plot(df["time"], df[col], label=col)
            if j == 3:
                axs[3, i].set_xticks(df["time"][::2]) 
                axs[3, i].set_xticklabels(df["time"][::2]) 
                axs[3, i].tick_params(axis='x', rotation=45)



    for ax in axs.flatten():
        ax.set_ylabel("Energy in $W/m^2$")
        ax.legend()

    plt.tight_layout()
    plt.savefig(PATH_PLOTS + 'diurnal_cycles/diurnal_cycles_comparison.png')


    # plot energy gaps
    plt.figure(figsize=(12, 4), dpi=600)
    for df, location in zip([df_bg, df_gw], ["BG", "GW"]):
        for alg, suffix in zip(['OG', 'MDS', 'MLP', 'RF'], ['_orig', '_f', '_f_mlp', '_f_rf']):
            if alg == "OG":
                label = location + ", Original Data"
                linestyle = '-'
            else:
                label = location +", Gapfilled " + alg
                linestyle = '--'
            plt.plot(df["time"], df["Q"] - df["G"] - df["H" + suffix] - df["LE" + suffix], label=label, linestyle=linestyle)
    plt.legend()
    plt.xticks(df_bg["time"][::2], rotation=45)
    plt.xlabel("Time")
    plt.ylabel("EBC gap in $W/m^2$")
    plt.tight_layout()
    plt.savefig(PATH_PLOTS + 'diurnal_cycles/energy_balances_closure.png')
    # plt.show()

    print(f"Saved figures to {PATH_PLOTS + 'diurnal_cycles/'}")

            


if __name__ == '__main__':
    plot_diurnal_cycles(path_gapfilled=PATH_GAPFILLED, path_plots=PATH_PLOTS)
