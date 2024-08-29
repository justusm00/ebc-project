import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.paths import PATH_GAPFILLED, PATH_PLOTS
plt.rcParams['font.size'] = 14



def plot_ebc(path_gapfilled, path_plots):
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

    # filter for day time
    df_bg = df_bg[(df_bg['time'] >= '10:00') &  (df_bg['time'] <= '17:00') ]
    df_gw = df_gw[(df_gw['time'] >= '10:00') &  (df_gw['time'] <= '17:00') ]



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
    df_gw["Res"] = df_gw["Q"] - df_gw["H_f_rf"] - df_gw["LE_f_rf"]
    df_bg["Res"] = df_bg["Q"] - df_bg["G"] - df_bg["H_f_rf"] - df_bg["LE_f_rf"]

    plt.figure()

    plt.plot(df_bg["time"], 100 * df_bg["Res"] / df_bg["Q"], label="BG")
    plt.plot(df_gw["time"], 100 * df_gw["Res"] / df_gw["Q"], label="GW")
    plt.xticks(df_bg["time"][::4], rotation=45)
    plt.xlabel("Time")
    plt.ylabel("EBC gap in percent")
    plt.tight_layout()
    plt.legend()

    plt.savefig(PATH_PLOTS + 'diurnal_cycles/ebc.pdf', dpi=150)




  
    print(f"Saved figures to {PATH_PLOTS + 'diurnal_cycles/'}")

            


if __name__ == '__main__':
    plot_ebc(path_gapfilled=PATH_GAPFILLED, path_plots=PATH_PLOTS)
