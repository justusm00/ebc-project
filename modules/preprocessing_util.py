import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from soil.soil import fill_thermal_conductivity, compute_soil_heatflux
from modules.util import transform_timestamp, numerical_to_float, get_day_of_year

from modules.columns import COLS_KEY




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
    
    # change bodentemp columns name for df_bg_23
    for idx in [1, 2, 3]:
        df_bg_23[f"Bodentemp_{idx}_30cm"] = df_bg_23[f"Bodentemp_30cm_{idx}"] 
        df_bg_23 = df_bg_23.drop(f"Bodentemp_30cm_{idx}", axis=1)

    # convert relevant columns to float
    df_bg_23_cols = [f"Bodentemp_{idx}_30cm"
                     for idx in [1, 2, 3] ] + ["Bodenwaermefluss",
                                                "kurzwEinstrahlung_300cm",
                                                "kurzwAusstrahlung_300cm",
                                                "Wasserdampfdefizit_200cm",
                                                "Wasserdampfdruck_200cm",
                                                "RelativeFeuchte_200cm",
                                                "Windgeschw_380cm",
                                                "Luftdruck",
                                                "Lufttemperatur_200cm"]
    df_gw_23_cols = [f"Bodentemp_{idx}_{depth}cm" for idx in [1, 2, 3]
                     for depth in [5, 15, 30]] + ["kurzwEinstrahlung_43m",
                                                "kurzwAusstrahlung_43m",
                                                "Luftdruck_43m",
                                                "Wasserdampfdefizit_43m",
                                                "Wasserdampfdruck_43m",
                                                "RelativeFeuchte_43m",
                                                "Windgeschw_I_43m",
                                                "langwEinstrahlung_43m",
                                                "langwAusstrahlung_43m",
                                                "Lufttemperatur_43m"]

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
    



    # get soil temperature for forest
    for idx in [1, 2, 3]:
        for depth in [5, 15, 30]:
            df_gw_23[f"soilTemperature_{idx}_{depth}cm"] = df_gw_23[f"Bodentemp_{idx}_{depth}cm"]
            df_gw_24[f"soilTemperature_{idx}_{depth}cm"] = df_gw_24[f"Bodentemp_{idx}_{depth}cm"]
            df_gw_24[f"soilMoisture_{idx}_{depth}cm"] = df_gw_24[f"Bodenfeuchte_{idx}_{depth}cm"]


    # get mean soil temperature (can be used as training feature)
    df_gw_23["soilTemperature"] = df_gw_23[[f"Bodentemp_{idx}_30cm" for idx in [1, 2, 3]]].mean(axis=1)
    df_gw_24["soilTemperature"] = df_gw_24[[f"Bodentemp_{idx}_30cm" for idx in [1, 2, 3]]].mean(axis=1)
    df_bg_23["soilTemperature"] = df_bg_23[[f"Bodentemp_{idx}_30cm" for idx in [1, 2, 3]]].mean(axis=1)
    df_bg_24["soilTemperature"] = df_bg_24[[f"Bodentemp_{idx}_30cm" for idx in [1, 2, 3]]].mean(axis=1)






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

    df_bg_23["airTemperature"] = df_bg_23["Lufttemperatur_200cm"]
    df_gw_23["airTemperature"] = df_gw_23["Lufttemperatur_43m"]
    df_bg_24["airTemperature"] = df_bg_24["Lufttemperatur_200cm"]
    df_gw_24["airTemperature"] = df_gw_24["Lufttemperatur_43m"]


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



def merge_data(df_fluxes, df_meteo, path_save):
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

    return df



def create_artificial_gaps(df):
    """Add new column artifical_gap onto df (0 indicates no gap, 1 indicates short gap, 2 indicates long gap, 3 indicates very long gap)

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # create temporary time and timestamp columns
    # Convert the '30min' column to a timedelta representing the minutes
    df['time'] = pd.to_timedelta(df['30min'] * 30, unit='m')

    # Create the datetime column by combining 'year', 'month', 'day' and 'time'
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day']]) + df['time']
    def generate_artificial_gaps(df, short_gap_ratio=0.20, long_gap_ratio=0.30, very_long_gap_ratio=0.50):
        df["artificial_gap"] = 0 # 0 for no gap, 1 for short gap, 2 for long gap, 3 for very long gap
        total_half_hours = len(df)
        short_gap_length = 48
        long_gap_length = 7 * 48
        very_long_gap_length = 30 * 48
        num_short_gaps = int(short_gap_ratio * 0.25 * total_half_hours / short_gap_length)
        num_long_gaps = int(long_gap_ratio * 0.25 * total_half_hours / long_gap_length)
        num_very_long_gaps = int(very_long_gap_ratio * 0.25 * total_half_hours / very_long_gap_length)
        applied_gaps = []
        # Function to apply gaps
        def apply_gaps(num_gaps, gap_length, gap_description, applied_gaps):
            for _ in range(num_gaps):
                # print(f"Trying to create gap of length {gap_length} ")
                i = 0
                while True and i < 100:
                    i = i + 1
                    start_index = np.random.choice(df.index[:-gap_length])
                    end_index = start_index + gap_length
                    if end_index > df.index[-1]:
                        continue
                    gap_range = pd.date_range(start=df.loc[start_index, 'timestamp'], 
                                            end=df.loc[end_index, 'timestamp'], 
                                            freq='30min')
                    gap_indices = df[df['timestamp'].isin(gap_range)].index
                    if all(idx not in applied_gaps for idx in gap_indices) and \
                    (df.loc[gap_indices, ['H_orig', 'LE_orig']].isna().all(axis=1).sum() / gap_length) < 0.5:
                        df.loc[gap_indices, "artificial_gap"] = gap_description
                        applied_gaps.extend(gap_indices)
                        break
            return applied_gaps
        
        # Apply short gaps (24h)
        applied_gaps = applied_gaps + apply_gaps(num_short_gaps, short_gap_length, 1, applied_gaps)
        # Apply long gaps (7 days)
        applied_gaps = applied_gaps +  apply_gaps(num_long_gaps, long_gap_length, 2, applied_gaps)
        # Apply very long gaps (30 days)
        applied_gaps = applied_gaps + apply_gaps(num_very_long_gaps, very_long_gap_length, 3, applied_gaps)
        
        return df
    # Apply the artificial gaps
    df_bg_with_gaps = generate_artificial_gaps(df[df["location"] == 0].copy().reset_index())
    df_gw_with_gaps = generate_artificial_gaps(df[df["location"] == 1].copy().reset_index())

    df_final = pd.concat([df_bg_with_gaps, df_gw_with_gaps]).reset_index(drop=True)
    # drop time and timestamp columns
    df_final = df_final.drop(['time', 'timestamp'], axis=1)

        
    return df_final
