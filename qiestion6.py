import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
        right = 0
        for i in range(len(y_true)):
            if (y_true[i]==y_pred[i]):
                right +=1
        accuracy = right / len(y_true)
        return accuracy
name = ['crx', 'data']
    # name = ['data']
for n in name:
        read_csv = pd.read_csv(n+"_change.csv")
        X = read_csv.iloc[:,:-1].to_numpy()
        y = read_csv.iloc[:,-1:].to_numpy()
        y = np.where(y == 0, -1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )
        p = SVC()
        # p = SVM(learning_rate=0.001, n_iters=1000)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)

        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,predictions))
        print('\n')
        print(classification_report(y_test,predictions))

        print(n + " dataset, Perceptron classification accuracy", accuracy(y_test, predictions))