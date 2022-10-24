import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
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
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - x_i* y_[idx]
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
    from tqdm import tqdm

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
        best = 0
        low_accuracy = 0
        for r in tqdm(np.arange(0.01, 0.99, 0.01)):
            p = SVM(lambda_param = r)
            p.fit(X_train, y_train)
            predictions = p.predict(X_test)
            score = accuracy(y_test, predictions)
            if score > low_accuracy:
                low_accuracy = score
                best = r

        print(n + " dataset, Perceptron classification accuracy", low_accuracy,'and best C is',best)


