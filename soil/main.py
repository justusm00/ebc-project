import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing_soil import prepare_data
from soil import fill_thermal_conductivity, compute_soil_heatflux, compute_soil_heatflux2

# read data
df_flux_fbg = pd.read_csv("soil/data_2024/EddyCovarianceData/eng/FBG_fluxes_30min_20240401_20240608_eng.csv")
df_flux_goew = pd.read_csv("soil/data_2024/EddyCovarianceData/eng/GoeW_fluxes_30min_20240401_20240608_eng.csv")
df_meteo_fbg = pd.read_csv("soil/data_2024/MeteorologicalData/eng/FBG_meteo_30min_20240401_20240608_eng.csv")
df_meteo_goew = pd.read_csv("soil/data_2024/MeteorologicalData/eng/GoeW_meteo_30min_20240401_20240608_eng.csv")

# relevant columns
cols_meteo_fbg = ["netRadiation_300cm", "soilHeatFlux"]
cols_meteo_goew = ["incomingLongwaveRadiation_43m", "incomingShortwaveRadiation_43m", "outgoingLongwaveRadiation_43m", 
                   "outgoingShortwaveRadiation_43m"] \
                    + [f"soilMoisture_{idx}_{depth}cm" for idx in [1, 2, 3] for depth in [5, 15, 30]]\
                    +[f"soilTemperature_{idx}_{depth}cm" for idx in [1, 2, 3] for depth in [5, 15, 30]]

cols_fluxes = ["H_f", "H_orig", "LE_f", "NEE_f"]

# prepare data
df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew = prepare_data(df_flux_fbg, df_flux_goew, df_meteo_fbg,
                                                                      df_meteo_goew, cols_fluxes, cols_meteo_fbg, cols_meteo_goew, "TIMESTAMP_START", fill_gaps=True)


# compute thermal conductivity
df_meteo_goew = fill_thermal_conductivity(df_meteo_goew)


# compute soil heatflux
df_meteo_goew = compute_soil_heatflux(df_meteo_goew)

def plot_diurnal_cycles_soil(df):
    df_agg = df.groupby("time").agg(G=("soilHeatflux", "mean"), moisture=("soilMoisture_1_5cm", "mean"), k=("thermalConductivity_5cm_mean", "mean"), dTdz = ("dTdz_mean", "mean"), T1_15cm = ("soilTemperature_1_15cm", "mean")).reset_index()
    plt.plot(df_agg["time"], df_agg["G"])
    plt.xticks(df_agg["time"].unique()[::2], rotation = 90)
    plt.ylabel("$G$")
    plt.tight_layout()
    plt.show()
    return df_agg

df_agg = plot_diurnal_cycles_soil(df_meteo_goew)