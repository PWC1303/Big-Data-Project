
#Cross validation and metrics imports
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np 
def run_cv(model,X_train,y_train,k =10):
    scaler = StandardScaler()
    kf = KFold(n_splits = k,shuffle = True,random_state = 33)
    score_ls = []
    k_count = 0
    for train_ind, test_ind in kf.split(X_train):
        k_count +=1
        X_tr, X_val = X_train[train_ind], X_train[test_ind]
        y_tr, y_val = y_train[train_ind], y_train[test_ind]
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        fit = model.fit(X_tr,y_tr)
        y_prob = fit.predict_proba(X_val)[:,1]
        roc_auc = roc_auc_score(y_val,y_prob)
        score_ls.append(roc_auc)

        print(f"Finished k = {k_count}")
    cv_score = np.mean(score_ls)
    return cv_score
   





