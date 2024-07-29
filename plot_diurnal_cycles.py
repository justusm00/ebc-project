import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.paths import PATH_GAPFILLED, PATH_PLOTS
plt.rcParams['font.size'] = 14



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

    # merge mds data
    df_mds = pd.read_csv(PATH_GAPFILLED + 'combined_gapfilled_mds.csv')
    df_mds["30min"] = df_mds["X30min"]
    df_mds["H_f_mds"] = df_mds["H_f"]
    df_mds["LE_f_mds"] = df_mds["LE_f"]
    df_mds.drop(["X30min", "H_f", "LE_f"], axis=1, inplace=True)
    df_bg_mds = df_mds[df_mds["location"] == 0]
    df_gw_mds = df_mds[df_mds["location"] == 1]
    df_bg_mds.drop("location", axis=1, inplace=True)
    df_gw_mds.drop("location", axis=1, inplace=True)

    df_bg = df_bg.merge(df_bg_mds, on=["30min", "year", "day_of_year"], how="outer")
    df_gw = df_gw.merge(df_gw_mds, on=["30min", "year", "day_of_year"], how="outer")


    # filter for 2024
    df_gw = df_gw[df_gw["year"] == 2024]
    df_bg = df_bg[df_bg["year"] == 2024]

    # add time column
    df_gw['time'] = df_gw['30min'].apply(lambda x: pd.Timedelta(minutes=30 * x))
    df_bg['time'] = df_bg['30min'].apply(lambda x: pd.Timedelta(minutes=30 * x))
    df_bg['time'] = df_bg['time'].apply(lambda x: f"{x.components.hours:02}:{x.components.minutes:02}")
    df_gw['time'] = df_gw['time'].apply(lambda x: f"{x.components.hours:02}:{x.components.minutes:02}")


    # add net radiation
    df_gw["Q"] = df_gw["netRadiation"]  
    df_bg["Q"] = df_bg["netRadiation"]


    def aggregate_data(df):
        df_agg = df.groupby("time").agg(G=("soilHeatflux", "mean"), H_orig = ("H_orig", "mean"), H_f_mds=("H_f_mds", "mean"),
                                        H_f_mlp=("H_f_mlp", "mean"), H_f_rf=("H_f_rf", "mean"),  LE_orig=("LE_orig", "mean"), LE_f_mds=("LE_f_mds", "mean"),
                                        LE_f_mlp=("LE_f_mlp", "mean"), LE_f_rf=("LE_f_rf", "mean"), Q=("Q", "mean")).reset_index()
        return df_agg


    df_gw = aggregate_data(df_gw)
    df_bg = aggregate_data(df_bg)




    # plot energy components and gaps
    fig, axs = plt.subplots(1, 2, figsize=(12,4), sharey=True)
    for df, location in zip([df_bg, df_gw], ["BG", "GW"]):
        for alg, suffix in zip(['OG', 'MDS', 'MLP', 'RF'], ['_orig', '_f_mds', '_f_mlp', '_f_rf']):
            if alg == "OG":
                label = location + ", Original Data"
                linestyle = '-'
            else:
                label = location +", Gapfilled " + alg
                linestyle = '--'
    axs[0].plot(df_bg["time"], df_bg["Q"], label="$Q_n$")
    axs[0].plot(df_bg["time"], df_bg["H_f_rf"], label="$H$")
    axs[0].plot(df_bg["time"], df_bg["LE_f_rf"], label="$\lambda E$")
    axs[0].plot(df_bg["time"], df_bg["Q"] - df_bg["G"] - df_bg["H_f_rf"] - df_bg["LE_f_rf"], label="Res")
    axs[0].plot(df_bg["time"], df_bg["G"], label="$G$")

    # leave out G for GW
    axs[1].plot(df_gw["time"], df_gw["Q"], label="$Q_n$")
    axs[1].plot(df_gw["time"], df_gw["H_f_rf"], label="$H$")
    axs[1].plot(df_gw["time"], df_gw["LE_f_rf"], label="$\lambda E$")
    axs[1].plot(df_gw["time"], df_gw["Q"] - df_gw["H_f_rf"] - df_gw["LE_f_rf"], label="Res")
    for ax in axs.flatten():
        ax.set_xticks(df_bg["time"][::4])
        ax.set_xticklabels(df_bg["time"][::4], rotation=45)
        ax.set_xlabel("Time")
        ax.grid()
        ax.legend()
    axs[0].set_ylabel("Energy in $W/m^2$")

    plt.xlabel("Time")
    plt.tight_layout()
    # plt.show()
    # plt.legend()
    plt.savefig(PATH_PLOTS + 'diurnal_cycles/energy_balances_closure.pdf', dpi=150)







    # plot only soil heatflux for gw
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    axs[1].plot(df_gw["time"], df_gw["G"])
    axs[0].plot(df_bg["time"], df_bg["G"])
    axs[0].set_ylabel("Soil heatflux $G$ in $W/m^2$")


    for ax in axs:
        ax.set_xticks(df_gw["time"][::4])
        ax.set_xticklabels(df_gw["time"][::4], rotation=45)        
        ax.set_xlabel("Time")
        ax.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(PATH_PLOTS + 'diurnal_cycles/soil_heatflux_both.pdf', dpi=150)
  


  
    print(f"Saved figures to {PATH_PLOTS + 'diurnal_cycles/'}")

            


if __name__ == '__main__':
    plot_diurnal_cycles(path_gapfilled=PATH_GAPFILLED, path_plots=PATH_PLOTS)
