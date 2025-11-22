from statsmodels.api import Logit
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,confusion_matrix
from sklearn.model_selection import train_test_split


def logm(y,X):
    X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.3)
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)




    model = Logit(y_train,X_train.astype(float)).fit()

    yhat_probs  = model.predict(X_test) 
    yhat = (yhat_probs>=0.5).astype(int)


    test_acc = accuracy_score(y_test,yhat)
    test_auc = roc_auc_score(y_test,yhat_probs)
    print(model.summary())

    print(f"Model accuracy on test set: {test_acc}")
    print(f"Model AUC{test_auc}")
