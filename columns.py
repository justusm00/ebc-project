# time and location columns, don't change this
COLS_TIME = ["year", "month", "day", "30min", "location"]
COLS_DAYOFYEAR = ["day_of_year"]

# columns used as labels for mlp training (nan filtering is done based on these)
COLS_LABELS = ["H_orig", "LE_orig"]

# columns used as features for mlp training 
# COLS_FEATURES = COLS_TIME + ["incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure",
#                  "waterPressureDeficit", "waterVaporPressure", "windSpeed"]
COLS_FEATURES = COLS_TIME + ["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit", "windSpeed"] # Results from importance analysis

# columns to keep from flux data
# allowed columns: "H_orig", "LE_orig", "H_f", "LE_f", 'CO2', 'H2O', 'Ustar', 'Reco'
COLS_FLUXES = COLS_TIME + ["H_orig", "LE_orig", "H_f", "LE_f"]

# columns to keep from meteo data
COLS_METEO = COLS_TIME + ["incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure",
              "waterPressureDeficit", "waterVaporPressure", "windSpeed", "relativeHumidity"]


