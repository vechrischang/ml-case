import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import DMatrix, XGBRegressor as XGB

def xgb_regressor(X, y):
    """
    learn_rate: 0.5-1
    eta: 0.01-0.2
    max_depth: 3-7, Used to control over-fitting as higher depth will allow model to learn relations 
    very specific to a particular sample.
    subsample: 0.5-1, Lower values make the algorithm more conservative and prevents overfitting but
    too small values might lead to under-fitting.
    """
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    xgb_clf = XGB(
        booster='gbtree',
        colsample_bytree=0.5,
        gamma=0.75,
        learning_rate=0.125,
        max_depth=10,
        min_child=8.0,
        n_estimators=400,
        subsample=0.9
    )
    
    xgb_clf.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae'
    )
    
    return xgb_clf
