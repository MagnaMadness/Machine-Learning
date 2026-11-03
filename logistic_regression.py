import numpy as np
import pandas as pd



def sigm(z):

    sigmoid = 1 / (1 + np.exp(-np.clip(z, -100, 100)))

    return sigmoid

# print(sigm(2))

def logreg_grad_des(X, y, learning_rate, iter):

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    XT = X.T

    for i in range(0, iter):

        f = sigm(X.dot(w))
        grad = (XT.dot((f - y))) / n_samples
        w = w - (learning_rate * grad)

    return w

w_logreg_grad_descent = logreg_grad_des(X_train, y_train, 0.01, 1000)
print(w_logreg_grad_descent)
