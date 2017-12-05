import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.optimize import fmin_l_bfgs_b
import Orange
from generic import *

X, y = IO.read('train.csv')
lam = 0.01
lr = LogicalRegression(lam)
ovo = OneVsOne(lr)
thetas = ovo(X, y)
Xt, ids = IO.read('test.csv', False)
# calculate results on test data
ovor = LogicalRegression.sigmoid(thetas, Xt)
res = OneVsOne.result(ovor, len(np.unique(y)))
IO.write(res, ids)