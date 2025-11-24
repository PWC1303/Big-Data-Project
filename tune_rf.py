from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier


import numpy as np 
import pandas as pd
import json
def main():
    df = pd.read_csv("data/data_cleaned.csv")
    df =df.dropna()

    y = np.array(df["churn"])
    X = df.drop(columns=["churn","customer_id"])
    X = np.array(X)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=33)




    
    param_dist = {
        "n_estimators":      [200, 400, 800],
        "max_depth":         [None, 8, 12, 16, 24],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 4, 8],
        "max_features":      ["sqrt", "log2", 0.3, 0.5],
        "bootstrap":         [True],
        "class_weight":      [None, "balanced"],  # if you have imbalance
    }

    rfc = RandomForestClassifier(n_jobs=-1,random_state=33)


    random_search = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_dist,
    n_iter=60,
    cv=3,
    scoring="roc_auc",   # or "roc_auc_ovr" if multiclass
    verbose=2,
    n_jobs=-1,
    random_state=33)
    random_search.fit(X_tr,y_tr)

    
   
    with open("rf_best_params.json", "w") as f:
        json.dump(random_search.best_params_, f)


if __name__ == "__main__":
    main()