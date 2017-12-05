import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.optimize import fmin_l_bfgs_b
import Orange

class IO:
    def read(fn, train=True):
        X = np.genfromtxt(fn, delimiter = ',', usecols = np.arange(0, 93), skiprows = 1, dtype = float)
        # converts first line into int and saves it
        ids = X[:, 0].copy().astype(int)
        # changes first line to ones
        X[:, 0] = np.ones(X.shape[0])
        if(train):
            y = np.genfromtxt(fn, delimiter = ',', usecols = {94}, skiprows = 1, dtype = str)
            # converts classes from string to float numbers
            yn = np.zeros(len(y))
            i = 0
            for el in y:
                yn[i] = float(el[6])
                i += 1   
            y = yn
            return X, y
        else:
            return X, ids
    def write(res, ids):
        fc = np.hstack((['id'], ids))
        oc = np.vstack((['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'], res))
        sres = np.column_stack((fc, oc))
        np.savetxt('results.csv', sres, delimiter=",", fmt="%s") 
class LogicalRegression:
    def __init__(self, lam):
        self.lam = lam
    def __call__(self, X, y):
        init_theta = np.zeros(X.shape[1]).T
        return fmin_l_bfgs_b(LogicalRegression.cost, init_theta, LogicalRegression.gradient, args=(X, y, self.lam))[0]
    def sigmoid(theta, x):
        return 1. / (1 + np.exp(-x.dot(theta)))
    
    #gradient for compution in l_bfsg
    def gradient(theta, X, y, lam):
        h = LogicalRegression.sigmoid(theta, X)
        g1 = (1. / X.shape[0]) * (h - y).T.dot(X[:, 0])
        g2 = (1. / X.shape[0]) * (((h - y).dot(X[:, 1:X.shape[1]])) + lam * theta[1:])
        return np.hstack((g1, g2)).T

    #cost function for compution in l_bfsg
    def cost(theta, X, y, lam):
        h = LogicalRegression.sigmoid(theta, X)
        return (1. / X.shape[0]) * ((-y.T.dot(np.log(h + 1e-10) )) - (1 - y.T).dot(np.log(1. - h + 1e-10))) + (lam / (2. * X.shape[0])) * (theta[1:].T.dot(theta[1:]))

class SoftmaxRegression:
    def __init__(self, lam):
        self.lam = lam
    def __call__(self, X, y):
        init_thetas=np.zeros(len(np.unique(y)) * X.shape[1])
        return fmin_l_bfgs_b(SoftmaxRegression.cost, init_thetas, SoftmaxRegression.gradient, args=(X, y, self.lam, np.unique(y)))[0]
    def cost(theta, X, y, lam, c):
        #pd = 1 / sum(np.exp(theta.dot(X.T)))
        #for k in c:
            
        #print (pd)
        #pd.shape
        #print (y)
        #return (np.array(y == 1.0).astype(int))
        t = theta.reshape(len(c), X.shape[1])
        h = np.exp(X.dot(t.T))
        p = h / np.sum(h, axis = 1)[:, None]
        # elements with ones and zeros if calculation is for them
        gt = np.array([np.array(y == el).astype(int) for el in c])
        
        # by = np.eye(len(c))[y.ravel().astype(int)]
        
        tc = (-np.sum(np.log(p) * gt.T) + lam * theta.dot(theta) / 2.) / X.shape[0]
        #co = gt.dot(np.log(p).T)
        #print (co)
        #tc = -(np.sum(co) / X.shape[1])
        #print (test)
        return tc
    
    def gradient(theta, X, y, lam, c):
        t = theta.reshape(len(c), X.shape[1])
        h = np.exp(X.dot(t.T))
        p = h / np.sum(h, axis = 1)[:, None]
        gt = np.array([np.array(y == el).astype(int) for el in c])
        grad = (X.T.dot(p - gt.T).T + lam * t) / X.shape[0]
        #grad = np.array(-(gt - p).dot(X) / X.shape[1])
        return grad.ravel()
    
class OneVsAll:
    def __init__(self, f):
        self.f = f
    def __call__(self, X, y):
        t = []
        # all classes
        c = np.unique(y)
        # changes specific class to 1 and others to 0
        for el in c:
            yn = y.copy()
            yn[y == el] = 1
            yn[y != el] = 0
            t = np.append(t, self.f(X, yn))
        return t.reshape((len(t) / X.shape[1]), X.shape[1])
    
class OneVsOne:
    def __init__(self, f):
        self.f = f
    def __call__(self, X, y):
        t = []
        # all classes
        c = np.unique(y)
        # changes specific class to 1 and others to 0
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                #print (j)
                yn = y.copy()
                # finds elements in array for this comparison for later vector/matrix cut
                cut = np.logical_or(yn == c[i], yn == c[j])
                # sets desirable elements to 0 or 1
                yn[y == c[i]] = 1
                yn[y == c[j]] = 0
                yn = yn[cut]
                Xn = X[cut]
                t = np.append(t, self.f(Xn, yn))
        #return t
        return t.reshape((len(t) / ((len(c) * (len(c) - 1)) / 2)), (len(c) * (len(c) - 1)) / 2)
    def result(ovor, l):
        res = np.zeros((ovor.shape[0], l))
        k = 0
        for i in range(l - 1):
            for j in range(i + 1, l):
                fc = ovor[:,k].copy()
                fc[ovor[:,k] > 0.5] = 1
                fc[ovor[:,k] <= 0.5] = 0
                lc = ovor[:,k].copy()
                lc[ovor[:,k] > 0.5] = 0
                lc[ovor[:,k] <= 0.5] = 1
                res[:,i] = res[:,i] + fc
                res[:,j] = res[:,j] + lc
                k += 1
        return res / ovor.shape[1]