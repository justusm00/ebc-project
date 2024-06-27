import numpy as np
import pandas as pd




def convert_timestamps(df):
    """Create time and date columns based on timestamp columns

    Args:
        df (_type_): _description_
    """
    df["DATE_START"] = pd.to_datetime(df["TIMESTAMP_START"].str.split(" ").str[0])
    df["TIME_START"] = df["TIMESTAMP_START"].str.split(" ").str[1]
    df["DATE_MID"] = pd.to_datetime(df["TIMESTAMP_MID"].str.split(" ").str[0])
    df["TIME_MID"] = df["TIMESTAMP_MID"].str.split(" ").str[1]
    df["DATE_END"] = pd.to_datetime(df["TIMESTAMP_END"].str.split(" ").str[0])
    df["TIME_END"] = df["TIMESTAMP_END"].str.split(" ").str[1]
    return df


def convert_dtypes_fluxes(df):
    """Convert relevant flux columns to float

    Args:
        df (_type_): _description_
    """
    df["H_f"] = df["H_f"].astype(float)
    df["H_orig"] = df["H_orig"].astype(float)
    df["LE_f"] = df["LE_f"].astype(float)
    df["NEE_f"] = df["NEE_f"].astype(float)
    return df



def convert_dtypes_meteo_fbg(df):
    """Convert relevant meteo columns to float (for FBG data)

    Args:
        df (_type_): _description_
    """
    df["netRadiation_300cm"] = df["netRadiation_300cm"].astype(float)
    df["soilHeatFlux"] = df["soilHeatFlux"].astype(float)
    return df



def convert_dtypes_meteo_goew(df):
    """Convert relevant meteo columns to float (for GOEW data)
    In this case the ground heat flux needs to be calculated manually so more colujmns are needed

    Args:
        df (_type_): _description_
    """

    df["incomingLongwaveRadiation_43m"] = df["incomingLongwaveRadiation_43m"].astype(float)
    df["incomingShortwaveRadiation_43m"] = df["incomingShortwaveRadiation_43m"].astype(float)
    df["outgoingLongwaveRadiation_43m"] = df["outgoingLongwaveRadiation_43m"].astype(float)
    df["outgoingShortwaveRadiation_43m"] = df["outgoingShortwaveRadiation_43m"].astype(float)
    df["soilMoisture_1_15cm"] = df["soilMoisture_1_15cm"].astype(float)
    df["soilMoisture_1_30cm"] = df["soilMoisture_1_30cm"].astype(float)
    df["soilMoisture_1_5cm"] = df["soilMoisture_1_5cm"].astype(float)
    df["soilMoisture_2_15cm"] = df["soilMoisture_2_15cm"].astype(float)
    df["soilMoisture_2_30cm"] = df["soilMoisture_2_30cm"].astype(float)
    df["soilMoisture_2_5cm"] = df["soilMoisture_2_5cm"].astype(float)
    df["soilMoisture_3_15cm"] = df["soilMoisture_3_15cm"].astype(float)
    df["soilMoisture_3_30cm"] = df["soilMoisture_3_30cm"].astype(float)
    df["soilMoisture_3_5cm"] = df["soilMoisture_3_5cm"].astype(float)
    df["soilTemperature_1_15cm"] = df["soilTemperature_1_15cm"].astype(float)
    df["soilTemperature_1_30cm"] = df["soilTemperature_1_30cm"].astype(float)
    df["soilTemperature_1_5cm"] = df["soilTemperature_1_5cm"].astype(float)
    df["soilTemperature_2_15cm"] = df["soilTemperature_2_15cm"].astype(float)
    df["soilTemperature_2_30cm"] = df["soilTemperature_2_30cm"].astype(float)
    df["soilTemperature_2_5cm"] = df["soilTemperature_2_5cm"].astype(float)
    df["soilTemperature_3_15cm"] = df["soilTemperature_3_15cm"].astype(float)
    df["soilTemperature_3_30cm"] = df["soilTemperature_3_30cm"].astype(float)
    df["soilTemperature_3_5cm"] = df["soilTemperature_3_5cm"].astype(float)
    return df



def prepare_data(df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew):
    # drop first row (only contains units)
    df_flux_fbg = df_flux_fbg.drop(0)
    df_flux_goew = df_flux_goew.drop(0)
    df_meteo_fbg = df_meteo_fbg.drop(0)
    df_meteo_goew = df_meteo_goew.drop(0)
    df_flux_fbg = convert_timestamps(df_flux_fbg)
    df_flux_goew = convert_timestamps(df_flux_goew)
    df_meteo_fbg = convert_timestamps(df_meteo_fbg)
    df_meteo_goew = convert_timestamps(df_meteo_goew)
    df_flux_fbg = convert_dtypes_fluxes(df_flux_fbg)
    df_flux_goew = convert_dtypes_fluxes(df_flux_goew)
    df_meteo_fbg = convert_dtypes_meteo_fbg(df_meteo_fbg)
    df_meteo_goew = convert_dtypes_meteo_goew(df_meteo_goew)
    return df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew




