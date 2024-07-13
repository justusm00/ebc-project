import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from modules.paths import PATH_PREPROCESSED

data = pd.read_csv(PATH_PREPROCESSED + 'data_merged_with_nans.csv')
data = data.dropna() # Drop all nans for this analyis
data = data.drop("netRadiation", axis=1) # drop netRadiation since it is highly correlated to incomingShortwaveRadiation
data.head()

# Do the analysis for H & LE -> Drop from data
y = data[['H_orig', 'LE_orig']]
X = data.drop(columns=['H_orig', 'LE_orig', 'H_f', 'LE_f', 'year', 'month', 'day', '30min'])
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

plt.figure()
sns.barplot(x='importance', y='feature', data=H_importance_df)
plt.title('Feature importance for H')
plt.show()

# Visualization of immportances for H
LE_importance = DtreeLE.feature_importances_

# Save importances in dataframe for sorted visualization
LE_importance_df = pd.DataFrame({'feature': feature_names, 'importance': LE_importance})
LE_importance_df = LE_importance_df.sort_values(by='importance', ascending=False)

plt.figure()
sns.barplot(x='importance', y='feature', data=LE_importance_df)
plt.title('Feature importance for LE')
plt.show()

# Visualization of importances for both combined
Comb_importance = np.mean([estimator.feature_importances_ for estimator in Mout.estimators_], axis=0) # Mean of feature importances

Comb_importance_df = pd.DataFrame({'feature': feature_names, 'importance': Comb_importance})
Comb_importance_df = Comb_importance_df.sort_values(by='importance', ascending=False)

plt.figure()
sns.barplot(x='importance', y='feature', data=Comb_importance_df)
plt.title('Feature importance for both fluxes combined')
plt.show()

# Subplot containing all

colors = sns.color_palette("tab10", len(feature_names)) # Some colors for better readability
color_mapping = {feature_names[i]: colors[i] for i in range(len(feature_names))}

fig, ax = plt.subplots(1,3, figsize=(18,6), sharex=True, sharey=True)
sns.barplot(x='importance', y='feature', data=Comb_importance_df, ax=ax[0], palette=color_mapping)
sns.barplot(x='importance', y='feature', data=H_importance_df, ax=ax[1], palette=color_mapping)
sns.barplot(x='importance', y='feature', data=LE_importance_df, ax=ax[2], palette=color_mapping)
ax[0].set_title("Immportances for both fluxes")
ax[1].set_title("Immportances for H")
ax[2].set_title("Immportances for LE")
plt.tight_layout()