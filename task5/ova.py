import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.optimize import fmin_l_bfgs_b
import Orange
from generic import *

X, y = IO.read('train.csv')
lam = 0.01
lr = LogicalRegression(lam)
ova = OneVsAll(lr)
thetas = ova(X, y)
Xt, ids = IO.read('test.csv', False)
# calculate results on test data
res = LogicalRegression.sigmoid(thetas.T, Xt)
IO.write(res, ids)

def numerical_grad(f, params, epsilon):
    num_grad = np.zeros(len(params))
    perturb = np.zeros(len(params))
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad

#X = np.array([[2,2],[1,3],[2,1],[4,1],[3,3],[2,4]])
#y = np.array([0]*3+[1]*3)
#X = np.random.rand(1000, 7)
#y = (np.random.rand(1000)>0.3).astype(int)
#X = np.vstack((np.ones(X.shape[0]), X.T)).T
init_theta = np.ones(X.shape[1])


ag = LogicalRegression.gradient(init_theta, X, y, 0.01)
ng = numerical_grad(lambda params: LogicalRegression.cost(params, X, y, 0.01), init_theta, 1e-4)
print(np.sum((ag - ng)**2))