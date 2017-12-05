import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.stats.stats import pearsonr
from scipy.optimize import fmin_l_bfgs_b
from sklearn import cross_validation, preprocessing
from collections import defaultdict
import Orange

class DataPrep:
    #descriptors = Orange.data.Table("molecular_descriptors_data.txt")
    def legend(fn):
        with open(fn) as f:
            for l in f:
                vals_legend = l.strip().split('\t')[6:]
                return vals_legend
            
    # read train set
    def fix_dilution(s):
        return s.replace('"', '').strip()

    def load_data(fn):
        readings = defaultdict(list)
        with open(fn) as f:
            vals_legend = next(f).strip().split('\t')[6:]
            for l in f:
                l = l.strip()
                t = l.split('\t')
                cid, dilution, vals, intensity = t[0], DataPrep.fix_dilution(t[4]), list(map(float, t[6:])), t[3]
                readings[cid, intensity, dilution].append(vals)
        for a,b in readings.items():
            readings[a] = np.array(b)
        return dict(readings)

    def mean_indv_notnan(data):
        means = []
        #average non-nan elements
        for vals in data.T:
            nonnan = vals[~np.isnan(vals)]
            means.append(np.mean(nonnan))
        return np.array(means)

    def load_data_mean_indv(fn):
        readings = DataPrep.load_data(fn)
        r2 = {}
        for a,b in readings.items():
            r2[a] = DataPrep.mean_indv_notnan(np.array(b))
        return r2

class IO:
    # get learning data
    def readData(fileName):
        test_set = []
        with open(fileName) as f:
            i = 0
            for l in f:
                cid, dilution = l.strip().split('\t')
                test_set = np.concatenate((test_set, [cid]))
                i += 1
            return test_set

    # testing success on learning data
    def printError(X, intensity, theta):
        final_learning_data = X.dot(theta)
        #print ("Error on learning data: " + str(np.sqrt(sum(np.power(final_learning_data - intensity, 2))/len(intensity))))

    def exportResults(readFileName, writeFileName, legend, results):
        with open(readFileName) as f:
            with open(writeFileName, "wt") as fo:
                i = 0
                for l in f:
                    cid, dilution = l.strip().split('\t')
                    for j in range(results.shape[0]):
                        fo.write("%s\t%s\t%f\n" % (cid, legend[j], results[j,i]))
                    i += 1

class PipeData:
    # returns lines with data
    def indexes_array(array, value):
        b = (array[:,0]==int(value))
        indexes = [i for i in range(len(b)) if b[i]]
        if indexes == []:
            print ("NOT GOOD")
            return indexes
        return array[indexes[0],:]

    def concatenate_array(descriptors, key, value_X, value_Y=[], value=[], train=False):
        res = PipeData.indexes_array(descriptors, key)
        if (not(np.isnan(res).any())):
            value_X = np.concatenate((value_X, res))
            value_Y = np.concatenate((value_Y, value))
        else:
            if (not(train)):
                value_X = np.concatenate((value_X, res))
                value_Y = np.concatenate((value_Y, value))
        return value_X, value_Y

    def create_train_set(train_set, descriptors):
        intensity_X = []
        intensity_Y = []
        rest_X = []
        rest_Y = []
        for key, value in train_set.items():
            
            if key[2] == "1/1,000":
                intensity_X, intensity_Y = PipeData.concatenate_array(descriptors, key[0], intensity_X, intensity_Y, [value[0]], True)
            if key[1] == "high ":
                rest_X, rest_Y = PipeData.concatenate_array(descriptors, key[0], rest_X, rest_Y, value[1:], True)

        intensity_X = intensity_X.reshape((len(intensity_X)/descriptors.shape[1]),descriptors.shape[1])
        rest_X = rest_X.reshape((len(rest_X)/descriptors.shape[1]),descriptors.shape[1])
        rest_Y = rest_Y.reshape((len(rest_Y)/20),20)
        return intensity_X, intensity_Y, rest_X, rest_Y

    def create_test_set(test_set, descriptors):
        test_X = []
        i = 0
        for key in test_set:
            i += 1
            test_X, nothing = PipeData.concatenate_array(descriptors, key, test_X)

        test_X = test_X.reshape((len(test_X)/descriptors.shape[1]),descriptors.shape[1])
        return test_X

# calculates a new theta
def calculate_theta(X, y):
    lr = Orange.regression.LassoRegressionLearner(alpha = 8) #linearna regresija L2 - lr je objekt, ki se ucije metoda, ki pricakuje podatke
    new_table = Orange.data.Table(X, y)
    model = lr(new_table)
    return model.skmodel.coef_

# calculates result
def calculate_result(theta, test_set, descriptors):
    return test_X.dot(theta)



train_set = DataPrep.load_data_mean_indv("TrainSet-hw2.txt")

# legend
legend = DataPrep.legend("TrainSet-hw2.txt")

descriptors = Orange.data.Table("molecular_descriptors_data.txt").X

intensity_X, intensity_Y, rest_X, rest_Y = PipeData.create_train_set(train_set, descriptors)

intensity_theta = calculate_theta(intensity_X, intensity_Y)
test_set = IO.readData("predict.txt")
test_X = PipeData.create_test_set(test_set, descriptors)

intensity_result = calculate_result(intensity_theta, test_X, descriptors)


rest_result = []
for i in range(rest_Y.shape[1]):
    rest_theta = calculate_theta(rest_X, rest_Y[:,i])
    rest_result = np.concatenate((rest_result, calculate_result(rest_theta, test_X, descriptors)))

rest_result = rest_result.reshape((len(rest_result)/len(intensity_result)),len(intensity_result))
results = np.vstack((intensity_result, rest_result))

IO.exportResults("predict.txt", "mean.txt", legend, results)