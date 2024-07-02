# time and location columns, don't change this
COLS_TIME = ["date", "year", "month", "day", "30min", "location"]

# columns used as labels for mlp training (nan filtering is done based on these)
COLS_LABELS = ["H_orig", "LE_orig"]

# columns used as features for mlp training 
COLS_FEATURES = COLS_TIME + ["incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure",
                 "waterPressureDeficit", "waterVaporPressure", "windSpeed"]

# columns to keep from flux data
# allowed columns: "H_orig", "LE_orig", "H_f", "LE_f", 'CO2', 'H2O', 'Ustar', 'Reco'
COLS_FLUXES = COLS_TIME + ["H_orig", "LE_orig", "H_f", "LE_f"]

# columns to keep from meteo data
COLS_METEO = COLS_TIME + ["incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure",
              "waterPressureDeficit", "waterVaporPressure", "windSpeed", "relativeHumidity"]



PATH = 'data/data_files/'