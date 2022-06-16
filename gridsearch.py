import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import time

from xgboost import XGBRFRegressor, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection      import permutation_importance

from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error

from sklearn import tree
from sklearn.tree import export_graphviz

from Plots import plot_hist
from Plots import Plot_TestPred

from wurlitzer import sys_pipes


dir      = "DATA/"
file1    = 'BHBH_Delay_Time.csv'
file2    = 'BHBH_Delay_Time_Shuffled.csv'

BHBH     = pd.read_csv(dir+file1)
BHBH.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

# shuffled dataset
shuffled = pd.read_csv(dir+file1)
shuffled.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
shuffled.head()


### Normalization of each column
feature_names = np.array(['Mass_0', 'Mass_1', 'Semimajor', 'Eccentricity', 'Z', 'alpha'])

features       = BHBH[feature_names]
features_shfld = shuffled[feature_names]

# mean normalization
features=(features-features.mean())/features.std()
features_shfld=(features_shfld-features_shfld.mean())/features_shfld.std()


X = features.to_numpy()
Y = np.log10(BHBH.Delay_Time).to_numpy()

X_test2 = features_shfld.to_numpy()
Y_test2 = np.log10(shuffled.Delay_Time).to_numpy()

# split train and test set (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)

param_grid = {
    'n_estimators': [250, 300, 350, 450, 500],
    'max_depth': [8, 9, 10, 11],
    'min_child_weight': [2, 3],
    'subsample': [.05, .1, .15, .65, .7, .75],
    'colsample_bytree': [None],
    'grow_policy': ['lossguide'],
    'booster':['gbtree']
}

rf = XGBRFRegressor(n_jobs = -1, verbosity = 10)

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv = 3, n_jobs = -1, verbose = 10)

# Fit the random search model
with sys_pipes():
    grid_search.fit(X_train, Y_train)

print("Test  R2 Score : %.2f"%grid_search.score(X_test, Y_test))
print("Train R2 Score : %.2f"%grid_search.score(X_train, Y_train))

print("Best Params : ", grid_search.best_params_)
print("Feature Importances : ")
pd.DataFrame([grid_search.best_estimator_.feature_importances_], columns=['Mass_0', 'Mass_1', 'Semimajor', 'Eccentricity', 'Z', 'alpha']).to_csv('xgboost_features.csv')
pd.DataFrame.from_dict(grid_search.cv_results_).to_csv('grid_search.csv')

print(pd.DataFrame.from_dict(grid_search.cv_results_).to_csv('grid_search.csv'))
