# columns used as labels for mlp training (nan filtering is done based on these)
COLS_LABELS = ["H_orig", "LE_orig"]
# columns used as features for mlp training (nan filtering is done based on these)
COLS_FEATURES = ["incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure",
                 "waterPressureDeficit", "waterVaporPressure", "windSpeed"]
# columns to keep from flux data
# allowed columns: "H_orig", "LE_orig", "H_f", "LE_f", 'CO2', 'H2O', 'Ustar', 'Reco'
COLS_FLUXES = ["H_orig", "LE_orig", "H_f", "LE_f"]
# columns to keep from meteo data
# allowed columns: "incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure", "waterPressureDeficit", "waterVaporPressure", "windSpeed", "relativeHumidity"
COLS_METEO = ["incomingShortwaveRadiation", "outgoingShortwaveRadiation", "soilHeatflux", "airPressure",
              "waterPressureDeficit", "waterVaporPressure", "windSpeed", "relativeHumidity"]


PATH = 'data/data_files/'