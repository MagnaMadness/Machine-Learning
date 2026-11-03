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

def direct_linreg(X, y):

    XT = X.T

    try:
        
        w = np.linalg.inv(XT.dot(X)).dot(XT).dot(y)

    except Exception as E:

        print(E)
        U = (0.01 * np.eye(X.shape[1])) + XT.dot(X)
        w = np.linalg.inv(U).dot(XT).dot(y)

    return w

w_direct = direct_linreg(X_train, y_train)
print(w_direct)

def grad_des_linreg(X, y, learning_rate, iter):

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    XT = X.T

    for i in range(0, iter):

        grad = (XT.dot((X.dot(w) - y))) / n_samples
        w = w - (learning_rate * grad)

    return w

w_gradient_descent = grad_des_linreg(X_train, y_train, 0.01, 1000)
print(w_gradient_descent)

def stoch_grad_des_linreg(X, y, learning_rate, iter):

    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for i in range(0, iter):

        for j in range(0, n_samples):

            X_j = X[j]
            y_j = y[j]

            grad = (X_j.dot(w) - y_j) * X_j
            w = w - (learning_rate * grad)

    return w

w_stoch_gradient_descent = stoch_grad_des_linreg(X_train, y_train, 0.01, 1000)
print(w_stoch_gradient_descent)

def apply_normalization(X, mean, std):
    X_new = (X-mean)/(std + 1e-8)
    return X_new

X_norm_validation = apply_normalization(X_raw_validation, mean, std)
X_norm_test = apply_normalization(X_raw_test, mean, std)
X_validation = add_intercept(X_norm_validation)
X_test = add_intercept(X_norm_test)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    
#hyperparameter optimization using X_validation
learning_rates = [0.001, 0.01, 0.05, 0.1]
best_lr = learning_rates[0]
best_val_mse = float('inf')

for lr in learning_rates:
    w_temp = grad_des_linreg(X_train, y_train, learning_rate=lr, iter=1000)

    y_val_pred = X_validation.dot(w_temp)
    val_mse = mean_squared_error(y_validation, y_val_pred)
    
    print(f"LR: {lr}, Validation MSE: {val_mse}")
    
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_lr = lr

print(f"Best Learning Rate selected: {best_lr}")

#Metrics using X_test
w_final = grad_des_linreg(X_train, y_train, learning_rate=best_lr, iter=1500)
y_test_pred = X_test.dot(w_final)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Final Test MSE: {test_mse}")
print(f"Final Test MAE: {test_mae}")
print(f"Final Test R2 Score: {test_r2}")


