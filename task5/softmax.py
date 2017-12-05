import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.optimize import fmin_l_bfgs_b
import Orange
from generic import *

X, y = IO.read('train.csv')
lam = 0.01
lr = SoftmaxRegression(lam)
thetas = lr(X, y)
t = thetas.reshape(len(np.unique(y)), X.shape[1])
Xt, ids = IO.read('test.csv', False)
# calculate results on test data
res = LogicalRegression.sigmoid(t.T, Xt)
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
#init_theta = np.zeros(X.shape[1])

#np.array([[1, 2],[3, 4]]).sum(axis=1)
#ag = SoftmaxRegression.gradient(init_theta, X, y, 0.01)
#ng = numerical_grad(lambda params: LogicalRegression.cost(params, X, y, 0.01), init_theta, 1e-4)
#print(np.sum((ag - ng)**2))

init_thetas=np.ones(len(np.unique(y)) * X.shape[1])

ag = SoftmaxRegression.gradient(init_thetas, X, y, 0.01, np.unique(y))
ng = SoftmaxRegression.cost(init_thetas, X, y, 0.01, np.unique(y))
ng = numerical_grad(lambda params: SoftmaxRegression.cost(params, X, y, 0.01, np.unique(y)), init_thetas, 1e-4)
print(np.sum((ag - ng)**2))