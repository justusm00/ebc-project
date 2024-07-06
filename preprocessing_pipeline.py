import os
import pandas as pd
from tqdm import tqdm

from soil.soil import fill_thermal_conductivity, compute_soil_heatflux
from modules.util import transform_timestamp, numerical_to_float, get_day_of_year

from columns import COLS_METEO, COLS_FLUXES, COLS_LABELS_ALL, COLS_FEATURES_ALL, COLS_KEY
from paths import PATH_RAW, PATH_PREPROCESSED, PATH_MODEL_TRAINING
from sklearn.model_selection import train_test_split




######## Script to automate data preprocessing and merging

def preprocess_flux_data(path_raw, path_save, cols):
    """Preprocess flux data - transform timestamp, convert numerical columns to float and keep only relevant columns.

    Args:
        path (_type_): _description_
        cols (_type_): Columns used for MLP training, must be numerical

    Returns:
        _type_: _description_
    """
    # exclude time and location columns from list of columns to be converted to float
    cols_to_convert = [col for col in cols if col not in COLS_KEY]

    # collect files to preprocess
    files = [f for f in os.listdir(path_raw) if 'fluxes' in f]


    data = []

    for f in tqdm(files):
        try: 
            df = pd.read_csv(path_raw+f, sep=',').drop(0)
        except pd.errors.ParserError: 
            # the 2024 files use ';' as separator and ',' as decimal separator
            df = pd.read_csv(path_raw+f, sep=';').drop(0)

        # location based on file name (files should be properly labelled with either BG or GW!)
        # one-hot encode the location: BG (botanical garden)==0, GW (Goettinger forest)==1
        df['location'] = 0 if 'BG' in f else 1

        df = transform_timestamp(df, 'TIMESTAMP_START')
        df = numerical_to_float(df, cols_to_convert)

        # keep only columns of interest
        df = df[cols]

        data.append(df.copy())


        print(f'{f} done')


    # combine the preprocessed data into single dataframe
    data_final = pd.concat(data, axis=0, ignore_index=True)


    data_final.to_csv(path_save + 'flux_data_preprocessed.csv', index=False)

    return data_final
    




def preprocess_meteo_data(path_raw, path_save, cols):
    """Preprocess meteo data - transform timestamp, convert numerical columns to float, rename columns to english and keep only relevant columns.

    Args:
        path (_type_): _description_
        cols (_type_): _description_

    Returns:
        _type_: _description_
    """

    df_bg_23 = pd.read_csv(f'{path_raw}BG_meteo_30min_20230101_20230801.csv', sep=',', na_values=['NaN']).drop(0) # BG meteo 2023
    df_gw_23 = pd.read_csv(f'{path_raw}GW_meteo_30min_20230101_20230801.csv', sep=',', na_values=['NaN']).drop(0) # GW meteo 2023
    df_bg_24 = pd.read_csv(f'{path_raw}BG_meteo_30min_20240401_20240701.csv', sep=';', na_values=['NaN']).drop(0) # BG meteo 2024
    df_gw_24 = pd.read_csv(f'{path_raw}GW_meteo_30min_20240401_20240701.csv', sep=';', na_values=['NaN']).drop(0) # GW meteo 2024

    df_bg_23 = df_bg_23.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)
    df_gw_23 = df_gw_23.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)
    df_bg_24 = df_bg_24.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)
    df_gw_24 = df_gw_24.drop(["TIMESTAMP_MITTE", "TIMESTAMP_ENDE"], axis=1)


    # fix spelling error in df_gw_23
    df_gw_23["kurzwAusstrahlung_43m"] = df_gw_23["kurzwAusstrahlun_43m"]
    df_gw_23 = df_gw_23.drop("kurzwAusstrahlun_43m", axis=1)

    # convert relevant columns to float
    df_bg_23_cols = ["Bodenwaermefluss", "kurzwEinstrahlung_300cm", "kurzwAusstrahlung_300cm",
                "Wasserdampfdefizit_200cm", "Wasserdampfdruck_200cm", "RelativeFeuchte_200cm", "Windgeschw_380cm", "Luftdruck"]
    df_gw_23_cols = [f"Bodentemp_{idx}_{depth}cm" for idx in [1, 2, 3] for depth in [5, 15, 30]]
    df_gw_23_cols.extend(["kurzwEinstrahlung_43m", "kurzwAusstrahlung_43m", "Luftdruck_43m",
                    "Wasserdampfdefizit_43m", "Wasserdampfdruck_43m", "RelativeFeuchte_43m",
                    "Windgeschw_I_43m", "langwEinstrahlung_43m", "langwAusstrahlung_43m"])

    df_bg_24_cols = df_bg_23_cols.copy()
    df_gw_24_cols = df_gw_23_cols.copy()
    # add soil moisture
    df_gw_24_cols.extend([f"Bodenfeuchte_{idx}_{depth}cm" for idx in [1, 2, 3] for depth in [5, 15, 30]])


    df_bg_23 = numerical_to_float(df_bg_23, df_bg_23_cols)
    df_gw_23 = numerical_to_float(df_gw_23, df_gw_23_cols)
    df_bg_24 = numerical_to_float(df_bg_24, df_bg_24_cols)
    df_gw_24 = numerical_to_float(df_gw_24, df_gw_24_cols)




    df_bg_23["incomingShortwaveRadiation"] = df_bg_23["kurzwEinstrahlung_300cm"]
    df_gw_23["incomingShortwaveRadiation"] = df_gw_23["kurzwEinstrahlung_43m"]
    df_bg_24["incomingShortwaveRadiation"] = df_bg_24["kurzwEinstrahlung_300cm"]
    df_gw_24["incomingShortwaveRadiation"] = df_gw_24["kurzwEinstrahlung_43m"]

    df_bg_23["outgoingShortwaveRadiation"] = df_bg_23["kurzwAusstrahlung_300cm"]
    df_gw_23["outgoingShortwaveRadiation"] = df_gw_23["kurzwAusstrahlung_43m"]
    df_bg_24["outgoingShortwaveRadiation"] = df_bg_24["kurzwAusstrahlung_300cm"]
    df_gw_24["outgoingShortwaveRadiation"] = df_gw_24["kurzwAusstrahlung_43m"]

    # longwave radiation only available for gw
    df_gw_23["incomingLongwaveRadiation"] = df_gw_23["langwEinstrahlung_43m"]
    df_gw_23["outgoingLongwaveRadiation"] = df_gw_23["langwAusstrahlung_43m"]
    df_gw_24["incomingLongwaveRadiation"] = df_gw_24["langwEinstrahlung_43m"]
    df_gw_24["outgoingLongwaveRadiation"] = df_gw_24["langwAusstrahlung_43m"]


    # net radiation
    df_bg_23["netRadiation"] = df_bg_23["Nettostrahlung_300cm"]
    df_bg_24["netRadiation"] = df_bg_24["Nettostrahlung_300cm"]
    df_gw_23["netRadiation"] = df_gw_23["incomingShortwaveRadiation"] + df_gw_23["incomingLongwaveRadiation"] \
                                    - df_gw_23["outgoingShortwaveRadiation"] - df_gw_23["outgoingLongwaveRadiation"] 
    df_gw_24["netRadiation"] = df_gw_24["incomingShortwaveRadiation"] + df_gw_24["incomingLongwaveRadiation"] \
                                    - df_gw_24["outgoingShortwaveRadiation"] - df_gw_24["outgoingLongwaveRadiation"] 
    




    for idx in [1, 2, 3]:
        for depth in [5, 15, 30]:
            df_gw_23[f"soilTemperature_{idx}_{depth}cm"] = df_gw_23[f"Bodentemp_{idx}_{depth}cm"]
            df_gw_24[f"soilTemperature_{idx}_{depth}cm"] = df_gw_24[f"Bodentemp_{idx}_{depth}cm"]
            df_gw_24[f"soilMoisture_{idx}_{depth}cm"] = df_gw_24[f"Bodenfeuchte_{idx}_{depth}cm"]


    # compute soil heatflux for df_gw_24 and df_gw_23
    df_gw_24 = fill_thermal_conductivity(df_gw_24)
    df_gw_24 = compute_soil_heatflux(df_gw_24)

    for idx in [1, 2, 3]:
        # just use mean thermal conductivity here
        df_gw_23[f"thermalConductivity_{idx}_5cm"] = df_gw_24[f"thermalConductivity_{idx}_5cm"].mean()

    df_gw_23 = compute_soil_heatflux(df_gw_23)
    df_bg_23["soilHeatflux"] = df_bg_23["Bodenwaermefluss"]
    df_bg_24["soilHeatflux"] = df_bg_24["Bodenwaermefluss"]

    df_bg_23["airPressure"] = df_bg_23["Luftdruck"]
    df_gw_23["airPressure"] = df_gw_23["Luftdruck_43m"]
    df_bg_24["airPressure"] = df_bg_24["Luftdruck"]
    df_gw_24["airPressure"] = df_gw_24["Luftdruck_43m"]

    df_bg_23["waterPressureDeficit"] = df_bg_23["Wasserdampfdefizit_200cm"]
    df_gw_23["waterPressureDeficit"] = df_gw_23["Wasserdampfdefizit_43m"]
    df_bg_24["waterPressureDeficit"] = df_bg_24["Wasserdampfdefizit_200cm"]
    df_gw_24["waterPressureDeficit"] = df_gw_24["Wasserdampfdefizit_43m"]

    df_bg_23["waterVaporPressure"] = df_bg_23["Wasserdampfdruck_200cm"]
    df_gw_23["waterVaporPressure"] = df_gw_23["Wasserdampfdruck_43m"]
    df_bg_24["waterVaporPressure"] = df_bg_24["Wasserdampfdruck_200cm"]
    df_gw_24["waterVaporPressure"] = df_gw_24["Wasserdampfdruck_43m"]

    df_bg_23["relativeHumidity"] = df_bg_23["RelativeFeuchte_200cm"]
    df_gw_23["relativeHumidity"] = df_gw_23["RelativeFeuchte_43m"]
    df_bg_24["relativeHumidity"] = df_bg_24["RelativeFeuchte_200cm"]
    df_gw_24["relativeHumidity"] = df_gw_24["RelativeFeuchte_43m"]

    df_bg_23["windSpeed"] = df_bg_23["Windgeschw_380cm"]
    df_gw_23["windSpeed"] = df_gw_23["Windgeschw_I_43m"]
    df_bg_24["windSpeed"] = df_bg_24["Windgeschw_380cm"]
    df_gw_24["windSpeed"] = df_gw_24["Windgeschw_I_43m"]


    # add location column
    df_bg_23["location"] = 0
    df_gw_23["location"] = 1
    df_bg_24["location"] = 0
    df_gw_24["location"] = 1

    # concat all dataframes
    df = pd.concat([df_bg_23, df_gw_23, df_bg_24, df_gw_24])

    # transform timestamp
    df = transform_timestamp(df, 'TIMESTAMP_START')

    # keep only relevant columns
    df = df[cols]


    # save as csv
    df.to_csv(path_save + 'meteo_data_preprocessed.csv', index=False)

    return df



def merge_data(df_fluxes, df_meteo):
    """Perform outer merge on flux and meteo data and save resulting df as csv.

    Args:
        df_fluxes (_type_): _description_
        df_meteo (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_meteo.merge(df_fluxes, how="outer", on=COLS_KEY)
    # add day_of_year_column
    df['day_of_year'] = df.apply(get_day_of_year, axis=1)
    # save as csv
    df.to_csv('data/preprocessed/data_merged_with_nans.csv', index=False)

    return df



def preprocessing_pipeline(path_raw, path_preprocessed, path_model_training, cols_fluxes, cols_meteo, cols_features, cols_labels, test_size=0.2, random_state=42):
    """Preprocessing pipeline to create dataset for Gapfilling MLP.

    Args:
        path (_type_): _description_
        cols_fluxes (_type_): _description_
        cols_meteo (_type_): _description_

    Returns:
        _type_: _description_

    """
    df_fluxes = preprocess_flux_data(path_raw, path_preprocessed, cols_fluxes)
    df_meteo = preprocess_meteo_data(path_raw, path_preprocessed, cols_meteo)
    df_merged = merge_data(df_meteo, df_fluxes)



    # keep only feature and label columns
    df_merged = df_merged[cols_features + cols_labels]

    

    # drop nan rows
    len_before = df_merged.__len__()
    df_merged.dropna(axis=0, how='any', inplace=True, ignore_index=True)
    na_removed = len_before - df_merged.__len__()
    print(f'\nRows removed because of NA: {na_removed}\n')

    # do train test split
    # Define features and target
    X = df_merged.drop(cols_labels, axis=1)  # Features
    y = df_merged[cols_labels]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # concatenate again
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)


    # save as csv
    df_train.to_csv(path_model_training + 'training_data.csv', index=False)
    df_test.to_csv(path_model_training + 'test_data.csv', index=False)

    return 


if __name__ == '__main__': 
    preprocessing_pipeline(path_raw=PATH_RAW, path_preprocessed=PATH_PREPROCESSED, 
                           path_model_training=PATH_MODEL_TRAINING,
                           cols_fluxes=COLS_FLUXES, cols_meteo=COLS_METEO,
                           cols_features=COLS_FEATURES_ALL, cols_labels=COLS_LABELS_ALL)