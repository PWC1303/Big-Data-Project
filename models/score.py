from sklearn.metrics import make_scorer
import numpy as np 


def negchurncost(y_test,yhat, cost_fn =400,cost_fp = 40):
    fn = np.sum((y_test==1)&(yhat==0))
    fp = np.sum((y_test==0)&(yhat==1))

    return -1*(fn*cost_fn+fp*cost_fp)  #buisness 101 we want to minimize costs 

cost_scorer = make_scorer(negchurncost)