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


    # filter for 2024
    df_gw = df_gw[df_gw["year"] == 2024]
    df_bg = df_bg[df_bg["year"] == 2024]

    # add time column
    df_gw['time'] = df_gw['30min'].apply(lambda x: pd.Timedelta(minutes=30 * x))
    df_bg['time'] = df_bg['30min'].apply(lambda x: pd.Timedelta(minutes=30 * x))
    df_bg['time'] = df_bg['time'].apply(lambda x: f"{x.components.hours:02}:{x.components.minutes:02}")
    df_gw['time'] = df_gw['time'].apply(lambda x: f"{x.components.hours:02}:{x.components.minutes:02}")
    df_bg['date'] = pd.to_datetime(df_bg[['year', 'month', 'day']]) 
    df_gw['date'] = pd.to_datetime(df_gw[['year', 'month', 'day']]) 

    # filter for day time
    df_bg = df_bg[(df_bg['time'] >= '08:00') &  (df_bg['time'] <= '17:00') ]
    df_gw = df_gw[(df_gw['time'] >= '08:00') &  (df_gw['time'] <= '17:00') ]



    # add net radiation
    df_gw["Q"] = df_gw["netRadiation"]  
    df_bg["Q"] = df_bg["netRadiation"]


    def plot_aggregated(col_agg, df_bg, df_gw):
        def aggregate_data(df):
            df_agg = df.groupby(col_agg).agg(G=("soilHeatflux", "mean"), H_orig = ("H_orig", "mean"), H_f_rf=("H_f_rf", "mean"),  LE_orig=("LE_orig", "mean"), LE_f_rf=("LE_f_rf", "mean"), Q=("Q", "mean")).reset_index()
            return df_agg


        df_gw = aggregate_data(df_gw)
        df_bg = aggregate_data(df_bg)
        df_gw["Res"] = df_gw["Q"] - df_gw["H_f_rf"] - df_gw["LE_f_rf"]
        df_bg["Res"] = df_bg["Q"] - df_bg["G"] - df_bg["H_f_rf"] - df_bg["LE_f_rf"]

        plt.figure()

        plt.plot(df_bg[col_agg], df_bg["Res"] / df_bg["Q"], label="BG")
        plt.plot(df_gw[col_agg], df_gw["Res"] / df_gw["Q"], label="GW")
        if col_agg == 'time':
            step_size = 4
        else:
            step_size = 8
            plt.ylim(-0.4, 0.6)
        plt.xticks(df_bg[col_agg][::step_size], rotation=45)
        plt.xlabel(col_agg)
        plt.ylabel("$Res/Q_n$")
        plt.tight_layout()
        plt.legend()
        plt.grid()
        plt.savefig(PATH_PLOTS + f'diurnal_cycles/ebc_{col_agg}.pdf', dpi=150)




    for col_agg in ["time", "date"]:
        plot_aggregated(col_agg, df_bg, df_gw)
    print(f"Saved figures to {PATH_PLOTS + 'diurnal_cycles/'}")

            


if __name__ == '__main__':
    plot_ebc(path_gapfilled=PATH_GAPFILLED, path_plots=PATH_PLOTS)
