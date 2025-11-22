from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def rfcm_cv()   :
    rfc = RandomForestClassifier()
    param_grid = {
    "n_estimators": [200, 500, 800],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"]
    }
    
    grid = GridSearchCV(rfc,param_grid,scoring="roc_auc",cv=10,n_jobs=1)

    return grid


