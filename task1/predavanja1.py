import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

#Podatki
np.random.seed(42)
x = np.random.random_sample(10)
y = 1 - (x-0.5)**2 + np.random.normal(0,0.1, len(x)) #neka dost random funkcija
plt.plot(x, y, "o");

X = np.column_stack([np.ones(len(x)), x]) #matrika X

def gradient_descent(X, y, alpha=0.01, epochs=1000):
    theta = np.zeros(X.shape[1]).T
    for i in range(epochs):
        theta = theta - alpha * (X.dot(theta) - y).dot(X)
    return theta

theta = gradient_descent(X, y)

mm = np.array((min(X[:, 1]), max(X[:, 1])))
P = np.column_stack((np.ones(len(mm)), mm))
P.dot(theta) #to je y
plt.plot(x, y, "o");
plt.plot(mm, P.dot(theta));

#ANALITICNA RESITEV - v DATAMININGU JE NE UPORABLJA NIHCE
def analitical(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)) #T - transponiraj, dot pomnozi

def poly_expand(X, k=9):
    for _ in range(k-1):
        X = np.hstack((X, np.multiply(X[:, 1], X[:, -1]).reshape(len(X), 1)))
    return X

poly_expand(X, 2)
poly_expand(X, 3)

poly_degree = 5
Z = poly_expand(X, poly_degree)
theta = analitical(Z, y)


plt.plot(Z[:, 1], y, "o")
xmin, xmax = Z[:, 1].min(), Z[:, 1].max()
xe = np.arrange(xmin, xmax (xmax -xmin)/100)
xe = np.vstack((np.ones(len(xe)) xe)).T
xe = poly_expand(xe, poly_degree)
plt.plot(xe[:, 1], xe.dot(theta), lw=2);

def regularized(X, y, r=0.1):
    L = np.eye(X.shape[1])
    L[0, 0] = 0 
    return np.linalg.inv(X.T.dot(X) + r * L).dot(X.T.dot(y))

regularized(Z, y, 0.1)