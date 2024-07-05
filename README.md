# Project goals

## Ground Heat Flux
- Determine thermal conductivity using relative moisture measurements, find value in literature [Van Wijk & de Vries]
- Gradients between all measurements (5->15, 15->30, 5->30) -> mean of gradients -> Ground heat flux
- Interpolation method: Diurnal means (Justus)


## Latent & Sensible Heat Flux
- Are included in the dataset, interpolation method?
- Neural Net? -> Use data from 2023 for additional training data -> discuss bias because of changing climate
- Data Cleaning -> Tag + Tageszeit codieren, NaNs entfernen, evtl. outlier & normalisieren (Dennis) 
- Predict H_orig and LE_orig (Robin)
- ( Data Augmentation )
- Compare to gap filling of original data and diurnal mean interpolation
- Normalization - use only trainset, but how to keep track of statistics for prediction?
- Use more features - also Meteo Data for gap filling!


## Results
- Mean daily energy balance closure over timespan of data, month
- Difference in balance closure between grass and forest (also over whole time & each month)
- Look for days with extreme gap values -> inspect metheo data and discuss


## ( Bowen Ratio method) 
- If we can't get to 15 minutes in our presentation :D 


# Code workflow for MLP gap filling

## Preprocessing

1. In columns.py, specify
    - columns used from flux data (COLS_FLUXES) and meteo data (COLS_METEO), all others are dropped
    - columns used as training features (COLS_FEATURES) and labels (COLS_LABELS), of course these must be included in COLS_FLUXES or COLS_METEO

2. Run script preprocessing_pipeline.py. This creates the following files: 
    - data_merged_with_nans.csv, flux_data_preprocessed.csv and meteo_data_preprocessed.csv under PATH_PREPROCESSED
    - test_data.csv and training_data.csv under PATH_MLP_TRAINING

3. Specify MLP parameters in MLP.py and run script. This trains the MLP and saves it under PATH_MODEL_SAVES. If normalization is True, the trainset statistics are also saved under PATH_MODEL_SAVES

4. Specify parameters in gap_filling.py and run the script. This creates two gapfilled datasets (one for GW and one for BG) under PATH_GAPFILLED. The model is automatically loaded according to the parameters specified. If normalization is true, the trainset statistics are automatically loaded.