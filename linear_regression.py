import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection import train_test_split, KFold, StratifiedFold

file_path = 'sample_data.csv'
np.random.seed(138)

def load_csv(file_path, target_column):

    df = pd.read_csv(file_path)

    X = df.drop(columns = [target_column]).values
    y = df[target_column].values

    return X, y

X_raw, y_raw = load_csv(file_path, 'Quality_Score')
# print(X_raw)
# print(y_raw)

def split_data(X, train_ratio, validation_ratio):

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

    return X_train, X_validation, X_test

X_raw_train, X_raw_validation, X_raw_test = split_data(X_raw, 0.7, 0.15)
# print(X_raw_train)
# print(X_raw_validation)
# print(X_raw_test)


def normalize_data(X):

    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    eps = 1e-8

    X_new = (X - mean) / (std + eps)

    return X_new

X_norm_train = normalize_data(X_raw_train)
# print(X_norm_train)

def add_bias(X):

    X_new = np.c_[np.ones(X.shape[0]), X]

    return X_new

X_train = add_bias(X_norm_train)
# print(X_train)

def direct_linreg(X, y):

    XT = X.T

    try:
        
        w = np.linalg.inv(XT.dot(X)).dot(XT).dot(y)

    except Exception as E:

        print(E)
        U = (0.01 * np.eye(X.shape[0])) + XT.dot(X)
        w = np.linalg.inv(U).dot(XT).dot(y)

    return w

w = direct_linreg(X_train, y_raw)
