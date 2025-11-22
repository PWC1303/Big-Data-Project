from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



def SVCm_cv():
    svc= SVC(probability=True)
    param_grid = {
        "C":[0.1,1,10,100],
        "kernel":["rbf","poly"],
        "gamma":["scale","auto"]
    }
   
    grid  =GridSearchCV(svc,param_grid,scoring="roc_auc",cv =10)

    return grid