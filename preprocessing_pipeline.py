from modules.preprocessing_util import preprocess_flux_data, preprocess_meteo_data, merge_data, create_artificial_gaps
from modules.paths import PATH_RAW, PATH_PREPROCESSED
from modules.columns import COLS_FLUXES, COLS_METEO, COLS_FEATURES_ALL, COLS_LABELS_ALL, COLS_KEY


def preprocessing_pipeline(path_raw, path_preprocessed, cols_fluxes, cols_meteo, 
                           cols_features, cols_labels):
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
    df_merged = merge_data(df_meteo, df_fluxes, path_save=path_preprocessed)

    df_merged = create_artificial_gaps(df_merged)

    # save as csv
    df_merged.to_csv(path_preprocessed + 'data_merged_with_nans.csv', index=False)


    return 


if __name__ == '__main__': 
    preprocessing_pipeline(path_raw=PATH_RAW, path_preprocessed=PATH_PREPROCESSED, 
                           cols_fluxes=COLS_FLUXES, cols_meteo=COLS_METEO,
                           cols_features=COLS_FEATURES_ALL, cols_labels=COLS_LABELS_ALL)