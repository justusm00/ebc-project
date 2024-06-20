import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import prepare_data
from soil import fill_thermal_conductivity

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
plt.figure()
plt.plot(df_meteo_goew["TIMESTAMP_START"], df_meteo_goew["thermalConductivity_1_5cm"], label = "5cm")
plt.plot(df_meteo_goew["TIMESTAMP_START"], df_meteo_goew["thermalConductivity_1_15cm"], label = "15cm")

plt.plot(df_meteo_goew["TIMESTAMP_START"], df_meteo_goew["thermalConductivity_1_30cm"], label = "30cm")


plt.xticks([])
plt.legend()
plt.show()
