import numpy as np 
import pandas as pd
import json
import joblib

import argparse
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from time import time



def main():

    def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)
    parser = argparse.ArgumentParser(prog='ProgramName',description='What the program does',epilog='Text at the bottom of help')
    parser.add_argument("--model", type=str, default="logm_l1")
    parser.add_argument("--cv", type = int, default=10)
    parser.add_argument("--poly",type=str,default="poly")
    args = parser.parse_args()
    df = pd.read_csv("data/data_cleaned.csv")
    df = df.dropna()
    y = np.array(df["churn"])
    X = np.array(df.drop(columns= ["churn","customer_id"]))
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=33)
    if args.model == "logm_l1":
        model = make_pipeline(StandardScaler(),LogisticRegressionCV(penalty="l1",solver ="liblinear",
                                cv =args.cv,scoring= "roc_auc" ,max_iter=1000,random_state=33,verbose=2))
        model.fit(X_tr,y_tr)
        results = model.named_steps["logisticregressioncv"]
        cv_auc = results.scores_[1].mean()
        params = results.get_params()
        best_c = float(results.C_[0])
        
        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params, "best_C": best_c})
        save_json(f"tuning_results/cv_auc/{args.model}_cv_auc.json", {"cv_auc": cv_auc})
        joblib.dump(model, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_auc and model.pkl saved for {args.model}")
    if args.model == "logm_l2":

        model = make_pipeline(StandardScaler(),
                              LogisticRegressionCV(penalty="l2",solver = "liblinear",
                                cv =args.cv,scoring="roc_auc" ,max_iter=1000,random_state=33,verbose=1))
      
        model.fit(X_tr,y_tr)
        results = model.named_steps["logisticregressioncv"]
        cv_auc = results.scores_[1].mean()
        params = results.get_params()
        best_c = float(results.C_[0])
        
        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params, "best_C": best_c})
        save_json(f"tuning_results/cv_auc/{args.model}_cv_auc.json", {"cv_auc": cv_auc})
        joblib.dump(model, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_auc and model.pkl saved for {args.model}")
    if args.model == "logm_ela":

        model =make_pipeline(StandardScaler(),LogisticRegressionCV(penalty="elasticnet",solver="saga",
                                l1_ratios=[x for x in np.linspace(0,1,10)],cv =args.cv,scoring="roc_auc" ,max_iter=1000,random_state=33,verbose =1))
      
        model.fit(X_tr,y_tr)
        results = model.named_steps["logisticregressioncv"]
        cv_auc = results.scores_[1].mean()
        params = results.get_params()
        best_c = float(results.C_[0])
        l1_ratio = float(results.l1_ratio_[0])
        
        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params, "best_C": best_c,"l1_ratio":l1_ratio})
        save_json(f"tuning_results/cv_auc/{args.model}_cv_auc.json", {"cv_auc": cv_auc})
        joblib.dump(model, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_auc and model.pkl saved for {args.model}")
    if args.model =="svc":
        model = make_pipeline(StandardScaler(),
                              SVC(probability=True))


        if args.poly:
            param_dist= json.load("svc_params_poly.json")
        if args.rbf:
            param_dist= json.load("svc_params_rbf.json")

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring="roc_auc",   
            n_jobs=-1,
            random_state=33,
            verbose=2)

        random_search.fit(X_tr,y_tr)

        params = random_search.best_params_
        cv_auc = random_search.best_score_
        

        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params})
        save_json(f"tuning_results/cv_auc/{args.model}_cv_auc.json", {"cv_auc": cv_auc})
        joblib.dump(random_search.best_estimator_, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_auc and model.pkl saved for {args.model}")

    if args.model =="rfc":
      param_dist = {
        "n_estimators":      [300, 400],
        "max_depth":         [None, 6, 12],
        "min_samples_leaf":  [1, 2, 4],
        "class_weight":      [None, "balanced"],
        "max_features":      ["sqrt"],   # fixed for 19 features, avoids noise
    }

      rfc = RandomForestClassifier(n_jobs=-1,random_state=33)
      random_search = RandomizedSearchCV(
      estimator=rfc,
      param_distributions=param_dist,
      n_iter=20,
      cv=3,
      scoring="roc_auc",   
      verbose=2,
      n_jobs=-1,
      random_state=33)
      random_search.fit(X_tr,y_tr)
      params = random_search.best_params_
      cv_auc = random_search.best_score_

      save_json(f"tuning_results/params/{args.model}_params.json",
        {"params": params})
      save_json(f"tuning_results/cv_auc/{args.model}_cv_auc.json", {"cv_auc": cv_auc})
      joblib.dump(random_search.best_estimator_, f"tuning_results/models/{args.model}.pkl")
      print(f"params, cv_auc and model.pkl saved for {args.model}")
if __name__ == "__main__":
    main()
       
        




