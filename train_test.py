
import pandas as pd 
import numpy as np
from Logistic_reg import logm
df = pd.read_csv("data/data_cleaned.csv")

print(df.head())
print(df.isna().sum())

df = df.dropna()

y = np.array(df["churn"])
print(f"number of churned customers in dataset: {np.sum( y==1)} \n which is around: {np.round((np.sum(y)/len(y))*100,2)}%  of customers ")

X = df.drop(columns=["churn","customer_id"])
logm(y,X)