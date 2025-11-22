
import pandas as pd 
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from CV import run_cv
from models.logistic import logm,logm_cv_l1,logm_cv_l2,logm_cv_ela
def main():

    parser = argparse.ArgumentParser(prog='ProgramName',description='What the program does',epilog='Text at the bottom of help')
    parser.add_argument("--model")
   
    df = pd.read_csv("data/data_cleaned.csv")
    df = df.dropna()

    y = np.array(df["churn"])
    print(f"number of churned customers in dataset: {np.sum( y==1)} \n which is around: {np.round((np.sum(y)/len(y))*100,2)}%  of customers ")
    X = df.drop(columns=["churn","customer_id"])
    X = np.array(X)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=33)
    
    
    logm_cv_score = run_cv(logm(),X_tr,y_tr)
    print(f"logistic regression cv_score {logm_cv_score}")
    logm_cv_l1_score= run_cv(logm_cv_l1(),X_tr,y_tr)
    print(f"logistic  l1 regression cv_score {logm_cv_l1_score}")
    logm_cv_l2_score= run_cv(logm_cv_l2(),X_tr,y_tr)
    print(f"logistic  l2 regression cv_score {logm_cv_l2_score}")
    logm_cv_ela_score= run_cv(logm_cv_ela(),X_tr,y_tr)
    print(f"logistic elastic-net regression cv_score {logm_cv_ela_score}")

    






main()