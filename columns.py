# time and location columns, don't change this, needed for merging !!
COLS_TIME = ["year", "month", "day", "30min", "location"]
# all possible columns that can be used as labels for mlp training (nan filtering is done based on these)
COLS_LABELS_ALL = ["H_orig", "LE_orig"]

# all possible columns tha can be used as features for mlp training (nan filtering is done based on these)
COLS_FEATURES_ALL = COLS_TIME + ["netRadiation", "incomingShortwaveRadiation", "soilHeatflux", "airPressure",
                 "waterPressureDeficit", "waterVaporPressure", "windSpeed", "relativeHumidity"]
# COLS_FEATURES = COLS_TIME + ["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit", "windSpeed"] # Results from importance analysis

# columns to keep from flux data
# allowed columns: "H_orig", "LE_orig", "H_f", "LE_f", 'CO2', 'H2O', 'Ustar', 'Reco'
COLS_FLUXES = COLS_TIME + ["H_orig", "LE_orig", "H_f", "LE_f"]

# columns to keep from meteo data
COLS_METEO = COLS_TIME + ["soilHeatflux", "airPressure", "incomingShortwaveRadiation", 
              "waterPressureDeficit", "waterVaporPressure", "windSpeed", "relativeHumidity", "netRadiation"]


