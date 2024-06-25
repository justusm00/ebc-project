import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import prepare_data
from soil import fill_thermal_conductivity, compute_soil_heatflux, compute_soil_heatflux2

# read data
df_flux_fbg = pd.read_csv("data_2024/EddyCovarianceData/eng/FBG_fluxes_30min_20240401_20240608_eng.csv")
df_flux_goew = pd.read_csv("data_2024/EddyCovarianceData/eng/GoeW_fluxes_30min_20240401_20240608_eng.csv")
df_meteo_fbg = pd.read_csv("data_2024/MeteorologicalData/eng/FBG_meteo_30min_20240401_20240608_eng.csv")
df_meteo_goew = pd.read_csv("data_2024/MeteorologicalData/eng/GoeW_meteo_30min_20240401_20240608_eng.csv")

# prepare data
df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew = prepare_data(df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew)


# compute thermal conductivity
df_meteo_goew = fill_thermal_conductivity(df_meteo_goew)
# df_meteo_goew.to_csv("meteo_with_conductivity.csv")

# visualize thermal conductivity
# plt.figure()
# plt.plot(df_meteo_goew["TIMESTAMP_START"], df_meteo_goew["thermalConductivity_1_5cm"], label = "5cm")
# plt.plot(df_meteo_goew["TIMESTAMP_START"], df_meteo_goew["thermalConductivity_1_15cm"], label = "15cm")

# plt.plot(df_meteo_goew["TIMESTAMP_START"], df_meteo_goew["thermalConductivity_1_30cm"], label = "30cm")


# plt.xticks([])
# plt.legend()
# plt.show()


# compute soil heatflux
df_meteo_goew = compute_soil_heatflux(df_meteo_goew)

def plot_diurnal_cycles_soil(df):
    df_agg = df.groupby("TIME_START").agg(G=("soilHeatflux", "mean"), moisture=("soilMoisture_1_5cm", "mean"), k=("thermalConductivity_5cm_mean", "mean"), dTdz = ("dTdz_mean", "mean"), T1_15cm = ("soilTemperature_1_15cm", "mean")).reset_index()
    plt.plot(df_agg["TIME_START"], df_agg["G"])
    plt.xticks(df_agg["TIME_START"].unique()[::2], rotation = 90)
    plt.ylabel("$G$")
    plt.tight_layout()
    plt.show()
    return df_agg

df_agg = plot_diurnal_cycles_soil(df_meteo_goew)