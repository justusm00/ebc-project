# key columns, don't change this, needed for merging !!
COLS_KEY = ["year", "month", "day", "30min", "location"]
# alternative key 
COLS_KEY_ALT = ["year", "day_of_year", "30min", "location"]
# all possible columns that can be used as labels for mlp training 
COLS_LABELS_ALL = ["H_orig", "LE_orig"]

# relevant meteo columns
COLS_METEO = COLS_KEY + ["soilHeatflux",
                         "airPressure",
                         "incomingShortwaveRadiation", 
                         "waterPressureDeficit",
                         "waterVaporPressure",
                         "windSpeed",
                         "relativeHumidity",
                         "netRadiation",
                         "airTemperature",
                         "soilTemperature"]

# all possible columns that can be used as features for mlp training
COLS_FEATURES_ALL = COLS_KEY+ ["day_of_year"] + COLS_METEO

# only important features
COLS_IMPORTANT_FEATURES = COLS_KEY + ["incomingShortwaveRadiation", "soilHeatflux", "waterPressureDeficit"]

# columns to keep from flux data
# allowed columns: "H_orig", "LE_orig", "H_f", "LE_f", 'CO2', 'H2O', 'Ustar', 'Reco'
COLS_FLUXES = COLS_KEY + ["H_orig",
                          "LE_orig",
                          "H_f",
                          "LE_f",
                          "CO2",
                          "Ustar"]


