from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd 

import numpy as np 



import json

with open("rf_best_params.json", "r") as f:
    best_params = json.load(f)
df = pd.read_csv("data/data_cleaned.csv")
df = df.dropna()
y = np.array(df["churn"])
X = np.array(df.drop(columns=["churn","customer_id"]))
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=33)

rfc = RandomForestClassifier(**best_params,n_jobs=-1,random_state= 33,verbose =1)

rfc.fit(X_tr,y_tr)

yhatprobs = rfc.predict_proba(X_te)[:, 1]

yhat = np.where(yhatprobs >=0.5, 1,0)

rf_acc = accuracy_score(y_te,yhat)

rf_roc_auc = roc_auc_score(y_te, yhatprobs)

print(f"Best rf model test accuracy {rf_acc}")
print(f"Best rf model roc_auc {rf_roc_auc}")