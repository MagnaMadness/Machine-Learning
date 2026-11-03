import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection import train_test_split, KFold, StratifiedFold

import numpy as np
import pandas as pd
file_path = 'sample_data.csv'
np.random.seed(138)

def load_csv(file_path, target_column):

    df = pd.read_csv(file_path)

    X = df.drop(columns = [target_column]).values
    y = df[target_column].values

    return X, y

X_raw, y_raw = load_csv(file_path, 'Quality_Score')


'''if we dont have a heading in the csv file, and its just 2 coulms, we can do this:
data = np.loadtxt(file_path, delimiter=",")
X = np.array(data[:, 0])
y = np.array(data[:, -1])

if there are 3 or more columns (X1, X2, X3, y... etc) then we can just:
data = np.loadtxt(file_path, delimiter=",")
X = np.array(data[:, :-1])
y = np.array(data[:, -1])
'''
# print(X_raw)
# print(y_raw)

def split_data(X, y, train_ratio, validation_ratio):

    num_datapoints = len(X)
    indices = np.random.permutation(num_datapoints)
    train_index_lim = int(train_ratio * num_datapoints)
    validation_index_lim = int((train_ratio + validation_ratio) * num_datapoints)

    train_idx = indices[:train_index_lim]
    validation_idx = indices[train_index_lim:validation_index_lim]
    test_idx = indices[validation_index_lim:]

    X_train = X[train_idx]
    X_validation = X[validation_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_validation = y[validation_idx]
    y_test = y[test_idx]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

X_raw_train, X_raw_validation, X_raw_test, y_train, y_validation, y_test = split_data(X_raw, y_raw, 0.7, 0.15)
# print(X_raw_validation)
# print(X_raw_test)


def normalize_data(X):

    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    eps = 1e-8

    X_new = (X - mean) / (std + eps)

    return X_new, mean, std

X_norm_train, mean, std = normalize_data(X_raw_train)
# print(X_norm_train)

def add_intercept(X):

    X_new = np.c_[np.ones(X.shape[0]), X]

    return X_new

X_train = add_intercept(X_norm_train)
# print(X_train)


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
