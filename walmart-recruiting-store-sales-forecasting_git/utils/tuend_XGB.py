import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import xgboost as xgb
from xgboost import DMatrix, XGBRegressor as XGB
from sklearn.metrics import mean_absolute_error
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin


def objective(space):
    
    clf = xgb.XGBRegressor(
        n_estimators = space['n_estimators'],
        max_depth = space['max_depth'],
        min_child_weight = space['min_child_weight'],
        subsample = space['subsample'],
        learning_rate = space['learning_rate'],
        gamma = space['gamma'],
        colsample_bytree = space['colsample_bytree'],
        objective='reg:linear'
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    clf.fit(X_train,
            y_train,
            eval_set=eval_set,
            eval_metric = 'mae')

    pred = clf.predict(X_val)
    mae = mean_absolute_error((y_val), (pred))

    return{'loss':mae, 'status': STATUS_OK }