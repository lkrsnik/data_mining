# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import Orange
from copy import copy
import sklearn
from sklearn import svm
import sklearn.metrics as skl_metrics
import zlib

# <codecell>

def cs (s):
    return len(zlib.compress(s))

def kernel (a, b):
    zlib_a = cs(a)
    zlib_b = cs(b)
    return ((cs(a + b) - zlib_a) / zlib_a) + ((cs(b + a) - zlib_b) / zlib_b)

def string_kernel(X, Y):
    R = []
    for x in X:
        lin = []
        for y in Y:
            i = int(x[0])
            j = int(y[0])
            # simplest kernel ever
            lin.append(1 / kernel (data[i], data[j]))
        R.append(np.array(lin))
    return np.array(R)

def read_files(location, start = 1, end = 21):
    eng = []
    for i in range(start, end):
        f = open(location + str(i), 'rb')
        eng.append(f.read())
        f.close()
    return np.array(eng)

def create_X(si, ei):
    return np.array(range(si, ei))[np.newaxis].T

def format_input(ad, c, d, X, y):
    data = np.concatenate((d, ad), axis=0)
    y = np.concatenate((y, np.zeros(ad.shape) + c), axis=0)
    X = np.concatenate((X, create_X(d.shape[0], d.shape[0] + ad.shape[0])), axis=0) if len(X) > 0 else create_X(d.shape[0], d.shape[0] + ad.shape[0])
    return data, X, y

# TEST IF SVM WORK ON LANGUAGES
"""eng = read_files('input/eng/')
slo = read_files('input/slo/')
data, X, y = np.array([]), np.array([]), np.array([])
data, X, y = format_input(eng[:-3], 0, data, X, y)
data, X, y = format_input(slo[:-3], 1, data, X, y)

X_t, y_t = np.array([]), np.array([])
data, X_t, y_t = format_input(eng[-3:], 0, data, X_t, y_t)
data, X_t, y_t = format_input(slo[-3:], 1, data, X_t, y_t)"""

# LYRICS TEST
"""queens = read_files('input/queens/')
keane = read_files('input/keane/')
nightwish = read_files('input/nightwish/')

data, X, y = np.array([]), np.array([]), np.array([])
data, X, y = format_input(queens[3:], 0, data, X, y)
data, X, y = format_input(keane[3:], 1, data, X, y)
data, X, y = format_input(nightwish[3:], 2, data, X, y)

X_t, y_t = np.array([]), np.array([])
data, X_t, y_t = format_input(queens[:3], 0, data, X_t, y_t)
data, X_t, y_t = format_input(keane[:3], 1, data, X_t, y_t)
data, X_t, y_t = format_input(nightwish[:3], 2, data, X_t, y_t)"""


# BASIC LYRICS TEST
"""ljubezenske = read_files('input/ljubezenske/')
otroske = read_files('input/otroske/')
napitnice = read_files('input/napitnice/')

data, X, y = np.array([]), np.array([]), np.array([])
data, X, y = format_input(ljubezenske[3:], 0, data, X, y)
data, X, y = format_input(otroske[3:], 1, data, X, y)
data, X, y = format_input(napitnice[3:], 2, data, X, y)

X_t, y_t = np.array([]), np.array([])
data, X_t, y_t = format_input(ljubezenske[:3], 0, data, X_t, y_t)
data, X_t, y_t = format_input(otroske[:3], 1, data, X_t, y_t)
data, X_t, y_t = format_input(napitnice[:3], 2, data, X_t, y_t)"""


# PROGRAMING LANGUAGE TEST
python = read_files('input/python/')
java = read_files('input/java/')
c = read_files('input/c/')

data, X, y = np.array([]), np.array([]), np.array([])
data, X, y = format_input(python[3:], 0, data, X, y)
data, X, y = format_input(java[3:], 1, data, X, y)
data, X, y = format_input(c[3:], 2, data, X, y)

X_t, y_t = np.array([]), np.array([])
data, X_t, y_t = format_input(python[:3], 0, data, X_t, y_t)
data, X_t, y_t = format_input(java[:3], 1, data, X_t, y_t)
data, X_t, y_t = format_input(c[:3], 2, data, X_t, y_t)


clf = svm.SVC(kernel=string_kernel)
clf.fit(X, y)
print(clf.predict(X_t))
print(y_t)

# <codecell>


