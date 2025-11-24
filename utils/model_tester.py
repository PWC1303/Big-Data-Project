from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,confusion_matrix,roc_curve
import json
import numpy as np
def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)
def model_tester(y_te,yhat_prob,model_name,alpha):


        yhat = np.where(yhat_prob>=alpha,1,0)

        test_acc = accuracy_score(y_te,yhat) 
        test_auc = roc_auc_score(y_te,yhat_prob)
        test_prescision = precision_score(y_te,yhat)
        test_f1 = f1_score(y_te,yhat)
        cm = confusion_matrix(y_te,yhat).tolist()

        file_name = f"testing_results/metrics/{model_name}_alpha_{alpha}_metrics_.json"
        save_json(f"{file_name}", 
                  {"alpha": alpha,
                   "testing_accuracy": test_acc,
                   "roc_auc": test_auc,
                   "precision":test_prescision,
                   "test_f1": test_f1,
                   "confusion_matrix": cm } )
        print(f"Metrics saved for {model_name} at {file_name}")