import os
import pandas as pd
from tqdm import tqdm

from soil.soil import fill_thermal_conductivity, compute_soil_heatflux
from modules.util import transform_timestamp, numerical_to_float

from columns import COLS_METEO, COLS_FLUXES, COLS_LABELS, COLS_FEATURES, COLS_TIME
from paths import PATH_RAW, PATH_PREPROCESSED, PATH_MLP_TRAINING, COLS_DAYOFYEAR
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
    cols_to_convert = [col for col in cols if col not in COLS_TIME+COLS_DAYOFYEAR]

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

    df1 = pd.read_csv(f'{path_raw}BG_meteo_30min_20230101_20230801.csv', sep=',', na_values=['NaN']).drop(0) # BG meteo 2023
    df2 = pd.read_csv(f'{path_raw}GW_meteo_30min_20230101_20230801.csv', sep=',', na_values=['NaN']).drop(0) # GW meteo 2023
    df3 = pd.read_csv(f'{path_raw}BG_meteo_30min_20240401_20240701.csv', sep=';', na_values=['NaN']).drop(0) # BG meteo 2024
    df4 = pd.read_csv(f'{path_raw}GW_meteo_30min_20240401_20240701.csv', sep=';', na_values=['NaN']).drop(0) # GW meteo 2024

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


    # add location column
    df1["location"] = 0
    df2["location"] = 1
    df3["location"] = 0
    df4["location"] = 1

    # concat all dataframes
    df = pd.concat([df1, df2, df3, df4])

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
    df = df_meteo.merge(df_fluxes, how="outer", on=COLS_TIME+COLS_DAYOFYEAR)
    # save as csv
    df.to_csv('data/preprocessed/data_merged_with_nans.csv', index=False)

    return df



def preprocessing_pipeline(path_raw, path_preprocessed, path_mlp_training, cols_fluxes, cols_meteo, cols_features, cols_labels, test_size=0.2, random_state=42):
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
    X = df_merged.drop(COLS_LABELS, axis=1)  # Features
    y = df_merged[COLS_LABELS]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # concatenate again
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)


    # save as csv
    df_train.to_csv(path_mlp_training + 'training_data.csv', index=False)
    df_test.to_csv(path_mlp_training + 'test_data.csv', index=False)

    return 


if __name__ == '__main__': 
    preprocessing_pipeline(path_raw=PATH_RAW, path_preprocessed=PATH_PREPROCESSED, path_mlp_training=PATH_MLP_TRAINING,
                           cols_fluxes=COLS_FLUXES+COLS_DAYOFYEAR, cols_meteo=COLS_METEO+COLS_DAYOFYEAR,
                           cols_features=COLS_FEATURES+COLS_DAYOFYEAR, cols_labels=COLS_LABELS+COLS_DAYOFYEAR)