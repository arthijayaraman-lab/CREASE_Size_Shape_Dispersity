# Training an XGBoost model to link structural features directly to the computed scattering profile
# Code written by the Jayaraman lab (2024) 

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import seaborn as sns
import skimage.metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from xgboost import DMatrix
import warnings
warnings.simplefilter('ignore')
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.utils import shuffle

#access the dataset 
df = pd.read_csv('train_dataset.csv')
df = df.dropna()
df_shuffled = shuffle(df, random_state=189)
X = df_shuffled.drop(columns=['sample id', 'I_q'])
y = df_shuffled['I_q']

#define the parameter space to get the optimized values using Bayesian optimization
param_space = {
    'n_estimators': np.arange(50, 1000, 50),
    'max_depth': np.arange(3, 15),
    'learning_rate': np.arange(0.001, 0.1, 0.001),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    'gamma': np.arange(0, 1, 0.1),
    'min_child_weight': np.arange(1, 10),
    'reg_lambda': np.arange(0.1, 1, 0.1),
    'reg_alpha': np.arange(0.1, 1, 0.1),
    'colsample_bylevel': np.arange(0.5, 1.0, 0.1)
}

#initialize the XGBoost model, for this work we use CPUs to train the XGBoost model.
xgb_reg = xgb.XGBRegressor(tree_method='hist', importance_type='cover', random_state=51)

#We use Skopt library to tune the parameter space
opt = BayesSearchCV(
    xgb_reg,
    param_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    return_train_score=True,
    refit=False,
    optimizer_kwargs={'base_estimator': 'GP'}
)


opt.fit(X, y)
best_params = opt.best_params_
best_score = opt.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

final_xgb = xgb.XGBRegressor(**best_params, tree_method='hist', importance_type='cover', random_state=51)
final_xgb.fit(X,y)

#get the weights assigned to each feature as cover method type
cover_importance = final_xgb.feature_importances_
print("Feature importance weights:", cover_importance)
#edit the path to save it to desired location
final_xgb.save_model('xgbmodel_ellipsoids.json')
