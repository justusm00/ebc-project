import numpy as np
import pandas as pd



def transform_timestamp(df, col_name):
    """
    Transform timestamp to proper date/year/month/day values
    
    Args:
        df (pandas dataframe): dataframe with timestamps
        col_name (str): column of original dataframe based on which to infer dates. Should be 'TIMESTAMP_START', 'TIMESTAMP_MITTE', or 'TIMESTAMP_ENDE'
    Returns:
        df (pandas dataframe)
    """

    df["date"] = pd.to_datetime(df[col_name].str.split(" ").str[0])
    df["time"] = df[col_name].str.split(" ").str[1]

    return df


def numerical_to_float(df, cols):
    """
    Args:
        df (pandas dataframe): dataframe to preprocess
        cols (list of str): names of columns to apply the dtype change to
    Returns:
        df (pandas dataframe)
    """
    for c in cols:
        try:
            df[f'{c}'] = df[f'{c}'].astype(dtype=float)
        except ValueError:
            # some files use ',' (comma) as decimal separator, replace with '.' (dot)
            df[f'{c}'] = df[f'{c}'].apply(lambda x: str(x).replace(',', '.'))
            df[f'{c}'] = df[f'{c}'].astype(dtype=float)
    
    return df


def interpolate_gaps(df, cols):
    """Gap fill relevant columns using simple linear interpolation
    Only suitable for small gaps 

    Args:
        df (_type_): _description_
        cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    for col in cols:
        df[col] = df[col].interpolate()

    return df



def prepare_data(df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew, cols_fluxes, cols_meteo_fbg, cols_meteo_goew, col_timestamps, fill_gaps=True):
    """Pipeline to prepare data for soil heatflux computation
    TODO add gap-filling

    Args:
        df_flux_fbg (_type_): _description_
        df_flux_goew (_type_): _description_
        df_meteo_fbg (_type_): _description_
        df_meteo_goew (_type_): _description_
        cols_fluxes (_type_): _description_
        cols_meteo_fbg (_type_): _description_
        cols_meteo_goew (_type_): _description_
        col_timestamps (_type_): _description_

    Returns:
        _type_: _description_
    """
    # drop first row (only contains units)
    df_flux_fbg = df_flux_fbg.drop(0)
    df_flux_goew = df_flux_goew.drop(0)
    df_meteo_fbg = df_meteo_fbg.drop(0)
    df_meteo_goew = df_meteo_goew.drop(0)
    # create date and time columns
    df_flux_fbg = transform_timestamp(df_flux_fbg, col_timestamps)
    df_flux_goew = transform_timestamp(df_flux_goew, col_timestamps)
    df_meteo_fbg = transform_timestamp(df_meteo_fbg, col_timestamps)
    df_meteo_goew = transform_timestamp(df_meteo_goew, col_timestamps)
    # convert relevant columns to float
    df_flux_fbg = numerical_to_float(df_flux_fbg, cols_fluxes)
    df_flux_goew = numerical_to_float(df_flux_goew, cols_fluxes)
    df_meteo_fbg = numerical_to_float(df_meteo_fbg, cols_meteo_fbg)
    df_meteo_goew = numerical_to_float(df_meteo_goew, cols_meteo_goew)
    # interpolate gaps in goew data
    if fill_gaps:
        df_meteo_goew = interpolate_gaps(df_meteo_goew, cols_meteo_goew)


    return df_flux_fbg, df_flux_goew, df_meteo_fbg, df_meteo_goew




