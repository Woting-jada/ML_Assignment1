import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.w)
                else:
                    self.w -= self.lr * (
                        2  * self.w - x_i* y_[idx]
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

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
        y = np.where(y <= 0, -1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123
        )
        p = SVM()
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
