import numpy as np
import pandas as pd
from preprocessing import prepare_data

# read data
df_flux_fbg = pd.read_csv("data_2024/EddyCovarianceData/eng/FBG_fluxes_30min_20240401_20240608_eng.csv")
df_flux_goew = pd.read_csv("data_2024/EddyCovarianceData/eng/GoeW_fluxes_30min_20240401_20240608_eng.csv")
df_meteo_fbg = pd.read_csv("data_2024/MeteorologicalData/eng/FBG_meteo_30min_20240401_20240608_eng.csv")
df_meteo_goew = pd.read_csv("data_2024/MeteorologicalData/eng/GoeW_meteo_30min_20240401_20240608_eng.csv")

# prepare data
df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew = prepare_data(df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew)

# plot soil moisture variation
plt.figure()
pl.t