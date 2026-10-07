import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("mnist_train.csv")
print(data.head())
data = np.array(data)
m, n = data.shape #m is the number of rows and n is the number of features plus 1 (because of the label column).
np.random.shuffle(data)
#Split the data to avoid overfitting

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

