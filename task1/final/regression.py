import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.stats.stats import pearsonr
from scipy.optimize import fmin_l_bfgs_b
from sklearn import cross_validation
import Orange

class CustomFunctions:
    #gradient for usage in l_bfsg
    def gradient(theta, X, y):
        return (1 / y.shape[0]) * ( (X.dot(theta) - y).dot(X))

    #cost function for usage in l_bfsg
    def costFunction(theta, X, y):
        return (1 / (2 * y.shape[0])) * ((X.dot(theta) - y).T.dot(X.dot(theta) - y))

    #gradient for usage in l_bfsg
    def gradient_regularization(theta, X, y):
        return (1 / y.shape[0]) * ( (X.dot(theta) - y).dot(X) + (lam / X.shape[1]) * theta)

    #cost function for usage in l_bfsg
    def costFunction_regularization(theta, X, y):
        return (1 / (2 * y.shape[0])) * ((X.dot(theta) - y).T.dot(X.dot(theta) - y) + lam * theta.T.dot(theta))

class IO:
    # get learning data
    def readData(fileName, setIntensity=False):
        if setIntensity:
            data = np.loadtxt(fileName, delimiter="\t", skiprows=3, usecols=(range(0,4870)))
        else:
            data = np.loadtxt(fileName, delimiter="\t", skiprows=3, usecols=(range(0,4869)))
        characteristics = data[:, 0:4869]
        if setIntensity:
            intensity = data[:, 4869]
            return characteristics, intensity
        return characteristics

    # testing success on learning data
    def printError(X, intensity, theta):
        final_learning_data = X.dot(theta)
        print ("Error on learning data: " + str(np.sqrt(sum(np.power(final_learning_data - intensity, 2))/len(intensity))))

    def exportResults(X, theta, fileName):
        file_ = open(fileName, 'w')
        for num in X.dot(theta):
            file_.write(str(num) + '\n')
        file_.close()

lam = 0.00003

def main():
    characteristics, intensity = IO.readData("train.tab.txt", True)
    indexes = getBestColumnsPearson(characteristics, intensity)
    theta1 = learnTheta(characteristics, intensity, indexes)
    theta = learnTheta(characteristics, intensity, indexes, CustomFunctions.costFunction_regularization, CustomFunctions.gradient_regularization)
    test_characteristics = IO.readData("test.tab")
    predictResults(theta1, test_characteristics, indexes, 'result')

#sets argument data and calculates theta using fmin_l_bfsg_b function
def learnTheta(characteristics, intensity, indexes, costFun=CustomFunctions.costFunction, gradientFun=CustomFunctions.gradient):
    #get arguments for l_bfsg
    bestcharact = characteristics[:, indexes]
    X = np.column_stack((np.ones(bestcharact.shape[0]), bestcharact))
    init_theta = np.zeros(X.shape[1]).T

    #calculation of theta (polinome)
    theta = fmin_l_bfgs_b(costFun, init_theta, gradientFun, args=(X, intensity))[0]

    #calculates if gradient is correct according to cost function
    original_gradient = gradientFun(theta, X, intensity)
    calculated_gradient = calculate_grad(costFun, X, intensity, theta, 1e-4)
    print(sum(np.power(original_gradient - calculated_gradient, 2)))

    #print testing success on learning data
    IO.printError(X, intensity, theta)
    return theta

def calculate_grad(costFun, X, intensity, theta, eps):
    grad = np.zeros(len(theta))
    temp = np.zeros(len(theta))
    for i in range(0, len(theta)):
        temp[i] = eps
        j1 = costFun((theta + temp), X, intensity)
        j2 = costFun((theta - temp), X, intensity)
        grad[i] = (j1 - j2) / (2. * eps)
        temp[i] = 0
    return grad

#calculates results and saves it into file
def predictResults(theta, characteristics, indexes, fileName):
    bestcharact = characteristics[:, indexes]
    X = np.column_stack((np.ones(bestcharact.shape[0]), bestcharact))
    IO.exportResults(X, theta, fileName)

#find columns with best corelation with pearson
def getBestColumnsPearson(characteristics, intensity):
    i = 0
    correlations = np.zeros(len(characteristics[0]))
    for column in characteristics.T:
        correlations[i] = pearsonr(intensity,column)[0]
        i += 1
    #minint = np.iinfo(np.int32).min
    correlations[np.isnan(correlations)] = 0
    return np.argpartition(correlations, -19)[-19:]

main()
