__author__ = 'jheaton'
# Based on the Python script used to collect the data for following paper/conference:
#
# Heaton, J. (2016, April). An Empirical Analysis of Feature Engineering for Predictive Modeling.
# In SoutheastCon 2016 (pp. 1-6). IEEE.
#
# http://www.jeffheaton.com

import math
import numpy as np
import time
import codecs
import csv
import multiprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

# Global parameters
TRAIN_SIZE = 10000
TEST_SIZE = 100
SEED = 2048
VERBOSE = 0
CYCLES = 5
FAIL_ON_NAN = False
NORMALIZE_ALL = False
DUMP_FILES = True
THREADS = multiprocessing.cpu_count()

# Another stopping class for deep learning.  If the error/loss falls
# below the specified threshold, we are done.


class AcceptLoss(object):
    def __init__(self, min=0.01):
        self.min = min

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']

        if current_valid < self.min:
            if VERBOSE > 0:
                print("Acceptable loss")
            raise StopIteration()


# Holds the data for the experiments.  This allows the data to be represented in
# the forms needed for several model types.
#
class DataHolder:
    def __init__(self, training, validation):
        self.X_train = training[0]
        self.X_validate = validation[0]
        self.y_train = training[1]
        self.y_validate = validation[1]

        # Provide normalized
        self.X_train_norm = MinMaxScaler().fit_transform(self.X_train)
        self.X_validate_norm = MinMaxScaler().fit_transform(self.X_validate)

        # Normalize all, if requested
        if NORMALIZE_ALL:
            self.X_train = StandardScaler().fit_transform(self.X_train)
            self.X_validate = StandardScaler().fit_transform(self.X_validate)

        # Format the y data for neural networks (lasange)
        self.y_train_nn = []
        self.y_validate_nn = []

        for y in self.y_train:
            self.y_train_nn.append([y])

        for y in self.y_validate:
            self.y_validate_nn.append([y])

        self.y_train_nn = np.array(self.y_train_nn, dtype=np.float32)
        self.y_validate_nn = np.array(self.y_validate_nn, dtype=np.float32)

    # Dump data to CSV files for examination.
    def dump(self, base):
        header = ",".join(["x" + str(x)
                           for x in range(1, 1 + self.X_train.shape[1])])
        header += ","
        header += ",".join(["y" + str(x)
                            for x in range(1, 1 + self.y_train_nn.shape[1])])

        np.savetxt(base + "_train.csv",
                   np.hstack((self.X_train, self.y_train_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")

        np.savetxt(base + "_validate.csv",
                   np.hstack((self.X_validate, self.y_validate_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")

        np.savetxt(base + "_train_norm.csv",
                   np.hstack((self.X_train_norm, self.y_train_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")

        np.savetxt(base + "_validate_norm.csv",
                   np.hstack((self.X_validate_norm, self.y_validate_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")


# Human readable time elapsed string.
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# Generate data for the provided function, usually a lambda.
def generate_data_fn(cnt, x_low, x_high, fn):
    return lambda rows: generate_data_fn2(rows, cnt, x_low, x_high, fn)


# Used internally for generate_data_fn
def generate_data_fn2(rows, cnt, x_low, x_high, fn):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        args = []
        for i in range(cnt):
            args.append(np.random.uniform(x_low, x_high))

        try:
            y = fn(*args)
            if not math.isnan(y):
                x_array.append(args)
                y_array.append(y)
        except (ValueError, ZeroDivisionError):
            pass

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Generate data for the ratio experiment
def generate_data_ratio(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        x = [
            np.random.uniform(0, 1),
            np.random.uniform(0.01, 1)]

        try:
            y = x[0] / x[1]
            x_array.append(x)
            y_array.append(y)
        except (ValueError, ZeroDivisionError):
            pass

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


def generate_data_poly2(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        x = [
            np.random.uniform(0, 1),
            np.random.uniform(0, 1)]

        y = 2 + 5*((x[0]*x[1]) ** 2) + (x[0]*x[1]*4)
        x_array.append(x)
        y_array.append(y)

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)

def generate_data_rpoly2(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        x = [
            np.random.uniform(0, 1),
            np.random.uniform(0, 1)]

        y = 1/(2 + 5*((x[0]*x[1]) ** 2) + (x[0]*x[1]*4))
        x_array.append(x)
        y_array.append(y)

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Generate data for the difference experiment
def generate_data_diff(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        x = [
            np.random.uniform(0, 1),
            np.random.uniform(0.01, 1)]

        try:
            y = x[0] - x[1]
            x_array.append(x)
            y_array.append(y)
        except (ValueError, ZeroDivisionError):
            pass

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Build a deep neural network for the experiments.
def neural_network_regression(data):
    model = Sequential()
    model.add(Dense(400, input_dim=data.X_train.shape[1], activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Grid-search for a SVM with good C and Gamma.
def svr_grid():
    param_grid = {
        'C': [1e-2, 1, 1e2],
        'gamma': [1e-1, 1, 1e1]

    }
    clf = GridSearchCV(SVR(kernel='rbf'), verbose=VERBOSE,
                       n_jobs=THREADS, param_grid=param_grid)
    return clf


# Perform an experiment for a single model type.
def eval_data(writer, name, model, data):
    model_name = model.__class__.__name__

    X_validate = data.X_validate
    X_train = data.X_train
    y_validate = data.y_validate
    y_train = data.y_train

    if model_name == "NeuralNet":
        y_validate = data.y_validate_nn
        y_train = data.y_train_nn
    elif model_name == "SVR":
        X_validate = data.X_validate_norm
        X_train = data.X_train_norm

    cycle_list = []
    v = 1 if VERBOSE > 0 else 0

    for cycle_num in range(1, CYCLES + 1):
        start_time = time.time()
        if 'KerasRegressor' in model_name:

            monitor = EarlyStopping(
                monitor='val_loss', min_delta=1e-3, patience=20, verbose=v, mode='auto')
            model.fit(data.X_train, data.y_train_nn,
                      validation_data=(data.X_validate, data.y_validate_nn),
                      callbacks=[monitor], verbose=v, epochs=100000)
        else:
            model.fit(X_train, y_train)
        elapsed_time = hms_string(time.time() - start_time)

        if 'KerasRegressor' in model_name:
            pred = model.predict(X_validate, verbose=v)
        else:
            pred = model.predict(X_validate)

        # Get the validatoin score
        if np.isnan(pred).any():
            if FAIL_ON_NAN:
                raise Exception("Unstable neural network. Can't validate.")
            score = 1e5
        else:
            score = np.sqrt(mean_squared_error(pred, y_validate))

        line = [name, model_name, score, elapsed_time]
        cycle_list.append(line)
        print("Cycle {}:{}".format(cycle_num, line))

    best_cycle = min(cycle_list, key=lambda k: k[2])
    print("{}(Best)".format(best_cycle))

    writer.writerow(best_cycle)


# Run an experiment over all model types
def run_experiment(writer, name, generate_data):
    np.random.seed(SEED)

    data = DataHolder(
        generate_data(TRAIN_SIZE),
        generate_data(TEST_SIZE))

    if DUMP_FILES:
        data.dump(name)

    # Define model types to use
    models = [
        svr_grid(),
        RandomForestRegressor(n_estimators=100),
        GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0, verbose=VERBOSE),
        KerasRegressor(build_fn=neural_network_regression, data=data)
    ]

    for model in models:
        eval_data(writer, name, model, data)


def main():
    with codecs.open("results.csv", "w", "utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(['experiment', 'model', 'error', 'elapsed'])
        start_time = time.time()
        run_experiment(writer, "diff", generate_data_diff)  # 1
        run_experiment(writer, "log", generate_data_fn(
            1, 1.0, 100.0, math.log))  # 2
        run_experiment(writer, "poly", generate_data_fn(
            1, 0.0, 2.0, lambda x: 1 + 5 * x + 8 * x ** 2))  # 3
        run_experiment(writer, "poly2", generate_data_poly2)  # 4
        run_experiment(writer, "pow", generate_data_fn(
            1, 1.0, 10.0, lambda x: x ** 2))  # 5
        run_experiment(writer, "ratio", generate_data_ratio)  # 6
        run_experiment(writer, "r_diff", generate_data_fn(
            4, 1.0, 10.0, lambda a, b, c, d: ((a - b) / (c - d))))  # 7
        run_experiment(writer, "r_poly", generate_data_fn(
            1, 1.0, 10.0, lambda x: 1 / (5 * x + 8 * x ** 2)))  # 8
        run_experiment(writer, "poly2", generate_data_rpoly2)  # 9
        run_experiment(writer, "sqrt", generate_data_fn(
            1, 1.0, 100.0, math.sqrt))  # 10

        elapsed_time = time.time() - start_time
        print("Elapsed time: {}".format(hms_string(elapsed_time)))


# Allow windows to multi-thread (unneeded on advanced OS's)
# See: https://docs.python.org/2/library/multiprocessing.html
if __name__ == '__main__':
    main()
