##### Script to run feature importance analysis for Random Forest


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from modules.paths import PATH_PREPROCESSED, PATH_PLOTS
from modules.columns import COLS_FEATURES_ALL
plt.rcParams['font.size'] = 14


# map feature names to feature symbols
feature_symbols = {"incomingShortwaveRadiation": "$R_{in,SW}$",
                   "location": "loc",
                   "soilTemperature": "$T_{soil}$",
                   "airTemperature": "$T_{air}$",
                   "day_of_year": "DoY",
                   "windSpeed": "$w$",
                   "30min": "30min",
                   "airPressure": "$P_{air}$",
                   "waterPressureDeficit": "$VPD$",
                   "relativeHumidity": "$rH$",
                   "year": "year"}

data = pd.read_csv(PATH_PREPROCESSED + 'data_merged_with_nans.csv')
data = data.dropna() # Drop all nans for this analyis
data = data.drop("netRadiation", axis=1) # drop netRadiation since it is highly correlated to incomingShortwaveRadiation
data = data.drop("waterVaporPressure", axis=1) # drop waterVaporPressure since who cares about it anyway

data.head()

# Do the analysis for H & LE -> Drop from data
y = data[['H_orig', 'LE_orig']]
X = data[[col for col in COLS_FEATURES_ALL if col not in ["day", "month", "soilHeatflux", "waterVaporPressure"]]]
X.head()

# Train-test-split and fitting the Decision trees
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Per flux
DtreeH = DecisionTreeRegressor(random_state=42)
DtreeLE = DecisionTreeRegressor(random_state=42)
DtreeH.fit(X_train, y_train['H_orig'])
DtreeLE.fit(X_train, y_train['LE_orig'])
# Fluxes combined
baseDtree = DecisionTreeRegressor(random_state=42)
Mout = MultiOutputRegressor(baseDtree)
Mout.fit(X_train, y_train)

# Visualization of immportances for H
H_importance = DtreeH.feature_importances_
feature_names = X.columns

# Save importances in dataframe for sorted visualization
H_importance_df = pd.DataFrame({'feature': feature_names, 'importance': H_importance})
H_importance_df = H_importance_df.sort_values(by='importance', ascending=False)


# Visualization of immportances for H
LE_importance = DtreeLE.feature_importances_

# Save importances in dataframe for sorted visualization
LE_importance_df = pd.DataFrame({'feature': feature_names, 'importance': LE_importance})
LE_importance_df = LE_importance_df.sort_values(by='importance', ascending=False)

# Visualization of importances for both combined
Comb_importance = np.mean([estimator.feature_importances_ for estimator in Mout.estimators_], axis=0) # Mean of feature importances

Comb_importance_df = pd.DataFrame({'feature': feature_names, 'importance': Comb_importance})
Comb_importance_df = Comb_importance_df.sort_values(by='importance', ascending=False)


# Assuming the DataFrames have 'feature' and 'importance' columns, add a new column to distinguish them
Comb_importance_df['type'] = 'Both Fluxes'
H_importance_df['type'] = '$H$'
LE_importance_df['type'] = '$\lambda E$'

# Combine the DataFrames
combined_df = pd.concat([Comb_importance_df, H_importance_df, LE_importance_df])

# add symbol name
combined_df["feature_symbol"] = combined_df["feature"].apply(lambda x: feature_symbols[x])

# Create the plot
plt.figure(figsize=(12, 8))
sns.barplot(y='importance', x='feature_symbol', hue='type', data=combined_df)
plt.xlabel("variable")
plt.ylabel("importance")
plt.legend(title=None)


# Set the title and layout
plt.tight_layout()

# Save the plot
plt.savefig(PATH_PLOTS + "importance/importance_all_combined.png", dpi=150)

# Show the plot
plt.show()