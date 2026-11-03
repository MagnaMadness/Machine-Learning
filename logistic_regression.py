import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file_path = 'data1_logistic.csv'
np.random.seed(67)

def load_csv(file_path, target_column):

    df = pd.read_csv(file_path)

    X = df.drop(columns = [target_column]).values
    y = df[target_column].values

    return X, y

data = np.loadtxt(file_path, delimiter=",")
X_raw = np.array(data[:, :-1])
y_raw = np.array(data[:, -1])

# X_raw, y_raw = load_csv(file_path, 'Quality_Score')
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
# print(w_logreg_grad_descent)



def apply_normalization(X, mean, std):
    X_new = (X-mean)/(std + 1e-8)
    return X_new

X_norm_validation = apply_normalization(X_raw_validation, mean, std)
X_norm_test = apply_normalization(X_raw_test, mean, std)
X_validation = add_intercept(X_norm_validation)
X_test = add_intercept(X_norm_test)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    
#hyperparameter optimization using X_validation (for logistic regression)

learning_rates = [0.001, 0.01, 0.05, 0.1]
best_lr = learning_rates[0]
best_val_acc = 0

for lr in learning_rates:

    w_temp = logreg_grad_des(X_train, y_train, learning_rate=lr, iter=1000)

    probabilities = sigm(X_validation.dot(w_temp))
    y_val_pred = np.where(probabilities >= 0.5, 1, 0)
    val_acc = np.mean(y_validation == y_val_pred)
    
    print(f"LR: {lr}, Validation Accuracy: {val_acc}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lr = lr

print(f"Best Learning Rate selected: {best_lr}")


w_final = logreg_grad_des(X_train, y_train, learning_rate=best_lr, iter=1500)

test_probabilities = sigm(X_test.dot(w_final))
y_test_pred = np.where(test_probabilities >= 0.5, 1, 0)

test_accuracy = np.mean(y_test == y_test_pred)
print(f"Final Test Accuracy: {test_accuracy}")


