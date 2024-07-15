# Project goals (need update!)

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


# Workflow for gap filling

## Install required libraries

To install the required python libraries, run
pip install -r requirements.txt

To install the required R packages, run
source("requirements.R") in RStudio

## Preprocessing

Run script preprocessing_pipeline.py. This creates the following files: 
- data_merged_with_nans.csv, flux_data_preprocessed.csv and meteo_data_preprocessed.csv under PATH_PREPROCESSED
- test_data.csv and training_data.csv under PATH_MODEL_TRAINING
The columns that should be gapfilled are assumed to have the format COL_NAME_orig. If you want to gapfill columns that don't end on _orig, you need to modify their column names in the preprocessing script.

## MLP Training

In MLP.py, you need to specify:
- features
- labels
- normalization T/F
- minmax_scaling T/F
- your initials (here you can also put any arbitrary string)
- further training / architecture params

When running the script, the model name is automatically created. The features and and labels are saved under model_saves/features and model_saves/labels, so that they can later be loaded for the gap filling. For each unique configuration of features/labels, a hash is created and added to the model name. This way, it is possible to easily train the same model architecture with different features / labels and use it directly for prediction without having to redefine anything in columns.py.

It is tried to load the train and test data for the given feature / label combination. If none is available, the train test split is performed and saved to data/mlp_training.

If normalization is set to True, the trainset statistics (mean and standard deviation) are saved to model_saves/mlp/statistics. It is important that the trainset statistics are used also to normalize the testset or new, unlabeled data. Similarly, minmax_scaling can be used instead of normalization. If both normalization and minmax_scaling are set to True, an error is raised since it does not make sense to use both.

TODO: find a way around hardcoding the params


## Random Forest Training

In RandomForest.py, you need to specify the features and labels that should be used for training. The features can be different from the ones used for the MLP, but the labels should be the same. The script fits the random forest to the training data using some hardcoded parameters (feel free to change). Similar to the MLP training, a hash is created based on the features and labels and the model is saved to model_saves/rf/ using this hash. 



## Gap filling

In gap_filling.py, you need to specify the filenames where the MLP and the RF are saved and the path to the preprocessed data. Everything else (features, labels, model architecture etc.) is determined from the model path.

Note that, for a specific row with gaps, the gaps can only be filled if all the features are not NaN. Since incomingShortwaveRadiation has almost no gaps, it makes sense to train an extra MLP / RF only on the key columns and incomingShortwaveRadiation to fill the remaining gaps. In this case you also need to specify filename_rfsw and filename_mlpsw. 

The script creates two files under data/gapfilled/ :

- BG_gapfilled.csv
- GW_gapfilled.csv

The data contains separate columns for gapfilling done with Marginal Distribution Sampling (-> the _f columns), MLP (-> the _f_mlp columns) and RF (-> the _f_rf columns).
