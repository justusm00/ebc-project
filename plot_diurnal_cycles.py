import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paths import PATH_GAPFILLED, PATH_PLOTS


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

    # plot bg data
    fig, axs = plt.subplots(4, 2, figsize =(20, 8))

    for ax in axs.flatten():
        ax.set_xticks([])  # Remove x-ticks
        ax.set_xticklabels([])

    axs[0, 0].set_title("GW, original data")
    axs[0, 0].plot(df_gw["time"], df_gw["Q"], label="Q")
    axs[0, 0].plot(df_gw["time"], df_gw["LE_orig"] + df_gw["H_orig"] + df_gw["G"], label="H + LE + G")

    axs[1, 0].set_title("GW, gapfilled data with MDS")
    axs[1, 0].plot(df_gw["time"], df_gw["Q"], label="Q")
    axs[1, 0].plot(df_gw["time"], df_gw["LE_f"] + df_gw["H_f"] + df_gw["G"], label="H + LE + G")


    axs[2, 0].set_title("GW, gapfilled data with MLP")
    axs[2, 0].plot(df_gw["time"], df_gw["Q"], label="Q")
    axs[2, 0].plot(df_gw["time"], df_gw["LE_f_mlp"] + df_gw["H_f_mlp"] + df_gw["G"], label="H + LE + G")


    axs[3, 0].set_title("GW, gapfilled data with Random Forest")
    axs[3, 0].plot(df_gw["time"], df_gw["Q"], label="Q")
    axs[3, 0].plot(df_gw["time"], df_gw["LE_f_rf"] + df_gw["H_f_rf"] + df_gw["G"], label="H + LE + G")
    axs[3, 0].set_xticks(df_gw["time"]) 
    axs[3, 0].set_xticklabels(df_gw["time"]) 
    axs[3, 0].tick_params(axis='x', rotation=90)


    axs[0, 1].set_title("BG, original data")
    axs[0, 1].plot(df_bg["time"], df_bg["Q"], label="Q")
    axs[0, 1].plot(df_bg["time"], df_bg["LE_orig"] + df_bg["H_orig"] + df_bg["G"], label="H + LE + G")

    axs[1, 1].set_title("BG, gapfilled data with MDS")
    axs[1, 1].plot(df_bg["time"], df_bg["Q"], label="Q")
    axs[1, 1].plot(df_bg["time"], df_bg["LE_f"] + df_bg["H_f"] + df_bg["G"], label="H + LE + G")


    axs[2, 1].set_title("BG, gapfilled data with MLP")
    axs[2, 1].plot(df_bg["time"], df_bg["Q"], label="Q")
    axs[2, 1].plot(df_bg["time"], df_bg["LE_f_mlp"] + df_bg["H_f_mlp"] + df_bg["G"], label="H + LE + G")



    axs[3, 1].set_title("BG, gapfilled data with Random Forest")
    axs[3, 1].plot(df_bg["time"], df_bg["Q"], label="Q")
    axs[3, 1].plot(df_bg["time"], df_bg["LE_f_rf"] + df_bg["H_f_rf"] + df_bg["G"], label="H + LE + G")
    axs[3, 1].set_xticks(df_gw["time"]) 
    axs[3, 1].set_xticklabels(df_gw["time"]) 
    axs[3, 1].tick_params(axis='x', rotation=90)

    for ax in axs.flatten():
        ax.set_ylabel("Energy in W/m^2")
        ax.legend()

    plt.tight_layout()
    plt.savefig(PATH_PLOTS + 'diurnal_cycles/diurnal_cycles_comparison.png')
    plt.show()





if __name__ == '__main__':
    plot_diurnal_cycles(path_gapfilled=PATH_GAPFILLED, path_plots=PATH_PLOTS)
