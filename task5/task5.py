# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import Orange
from sklearn import metrics
from scipy.optimize import fmin_l_bfgs_b

def g(z):
    return 1/(1+np.exp(-z))

def add_ones(X):
    return np.column_stack((np.ones(len(X)), X))

def logloss(res):
    ll = []
    for i in range(res.probabilities.shape[0]):
        # x je vektor verjetnosti, ki smo jih napovedali dejanskemu razredu
        x = np.array([v[i] for v, i in zip(res.probabilities[i], res.actual.astype(int))])
        ll.append(-sum(np.log(x))/len(x))
    return ll

# <codecell>

class IO:
    def read(fn, train=True):
        X = np.genfromtxt(fn, delimiter = ',', usecols = np.arange(0, 93), skiprows = 1, dtype = float)
        atts = [Orange.data.ContinuousVariable('x_' + str(num)) for num in range(93)]
        class_var = Orange.data.DiscreteVariable('class_', values=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        dbg_domain = Orange.data.Domain(attributes=atts, class_vars=class_var)
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
            return Orange.data.Table(dbg_domain, X, y)
        else:
            return X, ids
    def write(res, ids):
        fc = np.hstack((['id'], ids))
        oc = np.vstack((['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'], res))
        sres = np.column_stack((fc, oc))
        np.savetxt('results.csv', sres, delimiter=",", fmt="%s") 
class NeuralNetClassifier(Orange.classification.Model):
    """Neural network classifier based on a set of binary classifiers."""
    def __init__(self, domain, thetas):
        super().__init__(domain)
        self.thetas = thetas  # model parameters
    def predict(self, X):
        return self.h(X, self.thetas)[-1]

class NeuralNetLearner(Orange.classification.Learner):
    def __init__(self, arch, lambda_=1e-5):
        super().__init__()
        self.arch = arch
        self.lambda_ = lambda_
        self.name = "ann"

        self.theta_shape = np.array([(arch[i]+1, arch[i+1])
                                     for i in range(len(arch)-1)])
        ind = np.array([s1*s2 for s1, s2 in self.theta_shape])
        self.theta_ind = np.cumsum(ind[:-1])
        self.theta_len = sum(ind)

    def init_thetas(self, epsilon=0.01):
        return np.random.rand(self.theta_len) * 2 * epsilon - epsilon

    def shape_thetas(self, thetas):
        t = np.split(thetas, self.theta_ind)
        return [t[i].reshape(shape) for i, shape in enumerate(self.theta_shape)]

    def h(self, a, thetas):
        """feed forward, prediction"""
        thetas = self.shape_thetas(thetas)
        act = [a]
        for theta in thetas:
            a = g(add_ones(a).dot(theta))
            act.append(a)
        return act

    def cost_grad(self, thetas, X, Y):
        # use matrix and vector operations. could be written in a single line
        # use self.m as stored by the fit function
        ### COST ###
        shaped_thetas = self.shape_thetas(thetas)
        sumW = np.sum([np.sum(element[1:] ** 2) for element in shaped_thetas])
        h = self.h(X, thetas)
        cost = sum(sum((h[-1] - Y)**2)) / (2 * self.m) + ((self.lambda_ / (2 * self.m)) * sumW)
        ### GRAD ###
        pDelta = (-(Y - h[-1])) * (h[-1] * (1 - h[-1]))
        grad = []
        for theta, a in zip(reversed(shaped_thetas), reversed(h[:-1])):
            delta = pDelta.dot(theta[1:].T) * (a * (1 - a))
            theta[0] = np.zeros(theta.shape[1]).T
            g = (add_ones(a).T.dot(pDelta) + (self.lambda_ * theta)) / self.m
            grad = np.concatenate((g.flatten(), grad)) 
            pDelta = delta
        return cost, grad

    def grad_approx(self, thetas, X, Y, e=1e-1):
        return np.array([(self.cost_grad(thetas+eps, X, Y)[0] - self.cost_grad(thetas-eps, X, Y)[0])/(2*e)
                         for eps in np.identity(len(thetas)) * e])

    def reshapeY(self, Y):
        unique = np.unique(Y)
        # elements with ones and zeros if calculation is for them
        return np.array([np.array(Y == el).astype(float) for el in unique]).T
        
        
    def cost_grad_test(self, init_theta, X, Y):
        init_theta = self.init_thetas()
        ng = self.grad_approx(init_theta, X, Y)
        ag = self.cost_grad(init_theta, X, Y)[1]
        print(np.sum((ag - ng)**2))
        
    def fit(self, X, y, W=None):
        self.X, self.y = X, y
        self.m = self.X.shape[0]
        yn = self.reshapeY(y)
        
        thetas = self.init_thetas()
        # METODA KONCNIH RAZLIK
        print ("cost_grad_test")
        #print (self.cost_grad_test(thetas, X, yn))

        thetas, fmin, info = fmin_l_bfgs_b(self.cost_grad, thetas, args=(X, yn),
                                           factr=10)

        model = NeuralNetClassifier(self.domain, thetas)
        model.h = self.h
        return model

    def test(self, a):
        thetas = np.array([-30, 10, 20, -20, 20, -20, -10, 20, 20])
        print(self.h(a, thetas))
        

# <codecell>

# LOAD DATA
train = IO.read('train.csv')
test, ids = IO.read('test.csv', False)

# <codecell>

# IRIS TESTS
iris = Orange.data.Table("iris")
class_var = Orange.data.DiscreteVariable('c', values=(i for i in range(len(iris.domain.class_var.values))))
domain = Orange.data.Domain(iris.domain.attributes, class_var)
data = Orange.data.Table(domain, iris.X, iris.Y)
ann2 = NeuralNetLearner((4, 8, 3), lambda_=0.00001)
res2 = Orange.evaluation.CrossValidation(data, [ann2], k=10)
logloss(res2)

# <codecell>

# CROSS VALIDATION
ann3 = NeuralNetLearner((93,20,9), lambda_=0.00001)
res2 = Orange.evaluation.CrossValidation(train, [ann3], k=10)
logloss(res2)
# (93,20,9) - 2.3647389

# <codecell>

# GET RESULTS
ann = NeuralNetLearner((93,186,9), lambda_=0.00001)
model = ann(train)
res = model(test, ret=Orange.classification.Model.Probs)
IO.write(res, ids)

