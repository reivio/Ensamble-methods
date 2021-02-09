import numpy as np
from scipy.io import loadmat
mat = loadmat('data_train.mat')
mat2 = loadmat('data_test.mat')
X = np.array(mat["X"])   # train data predictors
y = mat["y"]   # train data labels
Xt = mat2["Xt"] # test data predictors
yt = mat2["yt"] # test data labels
y = np.reshape(y,(y.shape[0],))
yt = np.reshape(yt,(yt.shape[0],))