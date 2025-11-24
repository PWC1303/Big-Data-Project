from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
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
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)

    svc= SVC()
   
    param_dist = {
        "C":[0.1,1,10,100],
        "kernel":["rbf","poly"],
        "gamma":["scale","auto"],
        "degree":[2,3]
    }
    random_search = RandomizedSearchCV(
        estimator=svc,
        param_distributions=param_dist,
        n_iter=60,
        cv=3,
        scoring="roc_auc",   
        n_jobs=-1,
        random_state=33,
        verbose=2)

    random_search.fit(X_tr,y_tr)
       
    with open("svc_best_params.json", "w") as f:
        json.dump(random_search.best_params_, f)


if __name__ == "__main__":
    main()