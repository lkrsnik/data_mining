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
        return (1 / y.shape[0]) * (X.dot(theta) - y).dot(X)

    #cost function for usage in l_bfsg
    def costFunction(theta, X, y):
        return 1/(2 * y.shape[0]) * (X.dot(theta) - y).T.dot(X.dot(theta) - y)

    #gradient for usage in l_bfsg
    def gradient_regularization(theta, X, y):
        return ( (X.dot(theta) - y).dot(X) + (lam / X.shape[1]) * theta)
        #return (1 / y.shape[0]) * ((X.dot(theta) - y).dot(X)) + (lam / X.shape[1]) * theta
        #return ((X.dot(theta) - y).dot(X)+lam*theta)

    #cost function for usage in l_bfsg
    def costFunction_regularization(theta, X, y):
        #print (theta)
        #print (X)
        #print (y)
        #print (y.shape[0] == len (y))
        return (1/2) * ((X.dot(theta) - y).T.dot(X.dot(theta) - y)) + lam * theta.T.dot(theta)
        #return (1 / (2 * y.shape[0])) * (((X.dot(theta) - y).T.dot(X.dot(theta) - y)) + (lam * theta.T.dot(theta)))
        #return (1/2)*((X.dot(theta)-y).T.dot(X.dot(theta)-y))+lam*theta.dot(theta)

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

    

lam = 0.00000001

def main():
    characteristics, intensity = IO.readData("train.tab.txt", True)
    indexes = getBestColumnsPearson(characteristics, intensity)
    X = createMatrixOnData(characteristics, indexes)
    # -------------------------- CV ---------------------------------
    for i in range(0, 10):

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(characteristics, intensity, test_size=0.3, random_state=i)

        # ---------------------------------------------------------------

        #X, theta1 = learnTheta(characteristics, intensity, indexes)
        theta = learnTheta(X_train, y_train, indexes, CustomFunctions.costFunction_regularization, CustomFunctions.gradient_regularization)
        #IO.printError(X_test, y_test, theta)
    test_characteristics = IO.readData("test.tab")
    test_X = createMatrixOnData(test_characteristics, indexes)
    predictResults(theta, test_X, 'result')

def createMatrixOnData(characteristics, indexes):
    bestcharact = characteristics[:, indexes]
    return np.column_stack((np.ones(bestcharact.shape[0]), bestcharact))

#sets argument data and calculates theta using fmin_l_bfsg_b function
def learnTheta(X, intensity, indexes, costFun=CustomFunctions.costFunction, gradientFun=CustomFunctions.gradient):
    #get arguments for l_bfsg
    init_theta = np.zeros(X.shape[1]).T
    #calculation of theta (polinome)
    theta = fmin_l_bfgs_b(costFun, init_theta, gradientFun, args=(X, intensity))[0]


    # ----------------------------------- ERASE -----------------------------

    

    ag = gradientFun(init_theta, X, intensity)
    ng = numerical_grad(lambda params: costFun(params, X, intensity), init_theta, 1e-4)
    print(np.sum((ag - ng)**2))


    # -----------------------------------------------------------------------

    #print testing success on learning data
    IO.printError(X, intensity, theta)
    return theta

    # ----------------------------------- ERASE -----------------------------

def numerical_grad(f, params, epsilon):
    print (len(params))
    num_grad = np.zeros_like(params)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad

    # -----------------------------------------------------------------------

#calculates results and saves it into file
def predictResults(theta, X, fileName):
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

    #print (correlations.shape)
    #print (sum(correlations[np.argpartition(correlations, -20)[-20:]]))
    #print (sum(correlations[np.argpartition(correlations, 20)[:20]]))
    #print (np.hstack((np.argpartition(correlations, 10)[:10], np.argpartition(correlations, -10)[-10:])).shape)
    #print (correlations[1136])
    #indexes with highest correlation
    return np.argpartition(correlations, -20)[-20:]
    #indexes with lowest correlation
    #return np.argpartition(correlations, 20)[:20]
    #merged
    #return np.hstack((np.argpartition(correlations, 10)[:10], np.argpartition(correlations, -10)[-10:]))

#PCA = Orange.projection.PCA()(learning_characteristics)
#learning_characteristics = PCA(learning_characteristics)

#PCA.shape
main()


#21 -> 18.18437 