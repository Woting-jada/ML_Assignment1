import numpy as np


class VotedPerceptron:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.V = []
        self.C = []
        self.k = 0
    
    def fit(self, x, y):
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
        k = 0
        v = [np.ones_like(x)[0]]
        c = [0]
        for epoch in range(self.n_iter): 
            for i in range(len(x)):
                pred = 1 if np.dot(v[k], x[i]) > 0 else -1 # checks the sing of v*k
                if pred == y[i]: # checks if the prediction matches the real Y
                    c[k] += 1 # increments c
                else:
                    v.append(np.add(v[k], (y[i]*x[i])))
                    c.append(1)
                    k += 1
        self.V = v
        self.C = c
        self.k = k

    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for w,c in zip(self.V,self.C):
                s = s + c*np.sign(np.dot(w,x))
            preds.append(np.sign(1 if s>= 0 else 0))
        return preds

# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import pandas as pd

    def accuracy(y_true, y_pred):
        right = 0
        for i in range(len(y_true)):
            if (y_true[i]==y_pred[i]):
                right +=1
        accuracy = right / len(y_true)
        return accuracy
    name = ['crx', 'data']
    for n in name:
        read_csv = pd.read_csv(n+"_change.csv")
        X = read_csv.iloc[:,:-1].to_numpy()
        y = read_csv.iloc[:,-1:].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123
        )
        p = VotedPerceptron(n_iter=500)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)
        print(n + " dataset, Perceptron classification accuracy", accuracy(y_test, predictions))


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    # x0_1 = np.amin(X_train[:, 0])
    # x0_2 = np.amax(X_train[:, 0])

    # x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    # x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    # ymin = np.amin(X_train[:, 1])
    # ymax = np.amax(X_train[:, 1])
    # ax.set_ylim([ymin - 3, ymax + 3])

    # plt.show()
