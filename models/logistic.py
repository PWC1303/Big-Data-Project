from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,confusion_matrix
from sklearn.model_selection import train_test_split



def logm():
    return LogisticRegression(penalty="none", solver="lbfgs",
                              scoring = "roc_auc",max_iter=1000)

def logm_cv_l1():
    return LogisticRegressionCV(penalty="l1",solver ="liblinear",
                                cv =10,scoring="roc_auc" ,max_iter=1000)

def logm_cv_l2():
    return LogisticRegressionCV(penalty="l2",solver = "lbfgs",
                                cv =10,scoring="roc_auc" ,max_iter=1000)

def logm_cv_ela():
    return LogisticRegressionCV(penalty="elasticnet",solver="saga",
                                l1_ratios=[0.2, 0.5, 0.8],cv =10,scoring="roc_auc" ,max_iter=1000)