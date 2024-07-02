import pandas as pd
from modules.util import numerical_to_float
from soil.soil import fill_thermal_conductivity, compute_soil_heatflux
import matplotlib.pyplot as plt

###### Script to preprocess and merge meteo data
###### The files are quite heterogeneous (even though it may not seem so at first sight)
###### So this script is very tedious
###### Note that soil heatflux for GW for the year 2023 is computed based on the mean thermal conductivities in 2024, which may not be accurate
###### Also note that columns for BG and GW do not correspond to the same height, this information is lost here
###### But maybe the NN learns this just by the location encoding :))

df1 = pd.read_csv('data/data_files/BG_meteo_30min_20230101_20230801.csv', sep=',', na_values=['NaN']).drop(0) # BG meteo 2023
df2 = pd.read_csv('data/data_files/GW_meteo_30min_20230101_20230801.csv', sep=',', na_values=['NaN']).drop(0) # GW meteo 2023
df3 = pd.read_csv('data/data_files/BG_meteo_30min_20240401_20240608.csv', sep=';', na_values=['NaN']).drop(0) # BG meteo 2024
df4 = pd.read_csv('data/data_files/GW_meteo_30min_20240401_20240608.csv', sep=';', na_values=['NaN']).drop(0) # GW meteo 2024

df1 = df1.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)
df2 = df2.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)
df3 = df3.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)
df4 = df4.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)


# fix spelling error in df2
df2["kurzwAusstrahlung_43m"] = df2["kurzwAusstrahlun_43m"]
df2 = df2.drop("kurzwAusstrahlun_43m", axis=1)

# convert relevant columns to float
df1_cols = ["Bodenwaermefluss", "kurzwEinstrahlung_300cm", "kurzwAusstrahlung_300cm",
            "Wasserdampfdefizit_200cm", "Wasserdampfdruck_200cm", "RelativeFeuchte_200cm", "Windgeschw_380cm", "Luftdruck"]
df2_cols = [f"Bodentemp_{idx}_{depth}cm" for idx in [1, 2, 3] for depth in [5, 15, 30]]
df2_cols.extend(["kurzwEinstrahlung_43m", "kurzwAusstrahlung_43m", "Luftdruck_43m",
                 "Wasserdampfdefizit_43m", "Wasserdampfdruck_43m", "RelativeFeuchte_43m", "Windgeschw_I_43m"])

df3_cols = df1_cols.copy()
df4_cols = df2_cols.copy()
# add soil moisture
df4_cols.extend([f"Bodenfeuchte_{idx}_{depth}cm" for idx in [1, 2, 3] for depth in [5, 15, 30]])


df1 = numerical_to_float(df1, df1_cols)
df2 = numerical_to_float(df2, df2_cols)
df3 = numerical_to_float(df3, df3_cols)
df4 = numerical_to_float(df4, df4_cols)




df1["incomingShortwaveRadiation"] = df1["kurzwEinstrahlung_300cm"]
df2["incomingShortwaveRadiation"] = df2["kurzwEinstrahlung_43m"]
df3["incomingShortwaveRadiation"] = df3["kurzwEinstrahlung_300cm"]
df4["incomingShortwaveRadiation"] = df4["kurzwEinstrahlung_43m"]

df1["outgoingShortwaveRadiation"] = df1["kurzwAusstrahlung_300cm"]
df2["outgoingShortwaveRadiation"] = df2["kurzwAusstrahlung_43m"]
df3["outgoingShortwaveRadiation"] = df3["kurzwAusstrahlung_300cm"]
df4["outgoingShortwaveRadiation"] = df4["kurzwAusstrahlung_43m"]


for idx in [1, 2, 3]:
    for depth in [5, 15, 30]:
        df2[f"soilTemperature_{idx}_{depth}cm"] = df2[f"Bodentemp_{idx}_{depth}cm"]
        df4[f"soilTemperature_{idx}_{depth}cm"] = df4[f"Bodentemp_{idx}_{depth}cm"]
        df4[f"soilMoisture_{idx}_{depth}cm"] = df4[f"Bodenfeuchte_{idx}_{depth}cm"]


# compute soil heatflux for df4 and df2
df4 = fill_thermal_conductivity(df4)
df4 = compute_soil_heatflux(df4)

for idx in [1, 2, 3]:
    # just use mean thermal conductivity here
    df2[f"thermalConductivity_{idx}_5cm"] = df4[f"thermalConductivity_{idx}_5cm"].mean()

df2 = compute_soil_heatflux(df2)
df1["soilHeatflux"] = df1["Bodenwaermefluss"]
df3["soilHeatflux"] = df3["Bodenwaermefluss"]

df1["airPressure"] = df1["Luftdruck"]
df2["airPressure"] = df2["Luftdruck_43m"]
df3["airPressure"] = df3["Luftdruck"]
df4["airPressure"] = df4["Luftdruck_43m"]

df1["waterPressureDeficit"] = df1["Wasserdampfdefizit_200cm"]
df2["waterPressureDeficit"] = df2["Wasserdampfdefizit_43m"]
df3["waterPressureDeficit"] = df3["Wasserdampfdefizit_200cm"]
df4["waterPressureDeficit"] = df4["Wasserdampfdefizit_43m"]

df1["waterVaporPressure"] = df1["Wasserdampfdruck_200cm"]
df2["waterVaporPressure"] = df2["Wasserdampfdruck_43m"]
df3["waterVaporPressure"] = df3["Wasserdampfdruck_200cm"]
df4["waterVaporPressure"] = df4["Wasserdampfdruck_43m"]

df1["relativeHumidity"] = df1["RelativeFeuchte_200cm"]
df2["relativeHumidity"] = df2["RelativeFeuchte_43m"]
df3["relativeHumidity"] = df3["RelativeFeuchte_200cm"]
df4["relativeHumidity"] = df4["RelativeFeuchte_43m"]

df1["windSpeed"] = df1["Windgeschw_380cm"]
df2["windSpeed"] = df2["Windgeschw_I_43m"]
df3["windSpeed"] = df3["Windgeschw_380cm"]
df4["windSpeed"] = df4["Windgeschw_I_43m"]


cols = ["TIMESTAMP_START", "incomingShortwaveRadiation", "outgoingShortwaveRadiation",
        "soilHeatflux", "airPressure", "waterPressureDeficit", "waterVaporPressure", "relativeHumidity", "windSpeed"]

df1 = df1[cols]
df2 = df2[cols]
df3 = df3[cols]
df4 = df4[cols]


df1["Location"] = 0
df2["Location"] = 1
df3["Location"] = 0
df4["Location"] = 1


# concat all dataframes
df = pd.concat([df1, df2, df3, df4])

# save as csv
df.to_csv('data/meteo_data_preprocessed.csv')