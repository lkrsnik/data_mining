import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
from scipy.stats.stats import pearsonr
from scipy.optimize import fmin_l_bfgs_b
from sklearn import cross_validation, preprocessing
from collections import defaultdict
import Orange
from scoring import *

class DataPrep:
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

    def exportResults(writeFileName, ids, legend, results):
        results = results.clip(min=0)
        i = 0
        with open("input", "wt") as fi:
            with open(writeFileName, "wt") as fo:
                for l in ids:
                    cid = int(l[0])
                    fi.write("%s\t%s\n" % (cid, l[1]))
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

    def concatenate_array(descriptors, key, dilution, value_X, value_dilution, value_Y=[], value=[], train=False):
        res = PipeData.indexes_array(descriptors, key)
        if (not(np.isnan(res).any())):
            value_X = np.concatenate((value_X, res))
            value_Y = np.concatenate((value_Y, value))
            value_dilution = np.concatenate((value_dilution, [(key, dilution)])) if len(value_dilution) > 0 else [(key, dilution)]
        else:
            if (not(train)):
                value_X = np.concatenate((value_X, np.zeros(len(res))))
        return value_X, value_Y, value_dilution

    def create_train_set(train_set, descriptors):
        intensity_X = []
        intensity_Y = []
        intensity_dulition = np.array([])
        rest_X = []
        rest_Y = []
        rest_dulition = []
        for key, value in train_set.items():
            
            if key[2] == "1/1,000":
                intensity_X, intensity_Y, intensity_dulition = PipeData.concatenate_array(descriptors, key[0], key[2], intensity_X, intensity_dulition, intensity_Y, [value[0]], True)
                #intensity_dulition = np.concatenate((intensity_dulition, [(key[0], key[2])])) if len(intensity_dulition) > 0 else [(key[0], key[2])]
            if key[1] == "high ":
                rest_X, rest_Y, rest_dulition = PipeData.concatenate_array(descriptors, key[0], key[2], rest_X, rest_dulition, rest_Y, value[1:], True)
                #rest_dulition = np.concatenate((rest_dulition, [(key[0], key[2])])) if len(rest_dulition)>0 else [(key[0], key[2])]

        intensity_X = intensity_X.reshape((len(intensity_X)/descriptors.shape[1]),descriptors.shape[1])
        rest_X = rest_X.reshape((len(rest_X)/descriptors.shape[1]),descriptors.shape[1])
        rest_Y = rest_Y.reshape((len(rest_Y)/20),20)
        return intensity_X, intensity_Y, intensity_dulition, rest_X, rest_Y, rest_dulition

    def create_test_set(test_set, descriptors):
        test_X = []
        i = 0
        for key in test_set:
            i += 1
            test_X, nothing = PipeData.concatenate_array(descriptors, key, test_X)

        test_X = test_X.reshape((len(test_X)/descriptors.shape[1]),descriptors.shape[1])
        return test_X

def normalize_2(X):
    for i in range(X.shape[1]):
        if (sum(X[:, i]) > 0 and np.std(X[:, i]) > 0.00001):
            #print (sum(X[:, i]))
            X[:, i]=X[:, i]-(sum(X[:, i])/X.shape[1])/np.std(X[:, i])
            if np.isinf(X[:, i]).any():
                print (i)
                print (np.std(X[:, i]))
                print (X[:, i])
        else:
            X[:, i]=np.zeros(len(X[:, i]))
    return X

# calculates a new theta
def calculate_model(f, X, y, alpha):
    #lr = Orange.regression.LassoRegressionLearner(alpha = alpha, max_iter = 1000, normalize = False) 
    #linearna regresija L2 - lr je objekt, ki se ucije metoda, ki pricakuje podatke
    #y=y-(sum(y)/(len(y)))/np.std(y)
    lr = f(alpha) #linearna regresija L2 - lr je objekt, ki se ucije metoda, ki pricakuje podatke
    new_table = Orange.data.Table(X, y)
    
    model = lr(new_table)
    return model


# where the magic happens
def magic(descriptors, train_set, test_set, alpha):
    # formats train and test set
    intensity_X, intensity_Y, rest_X, rest_Y = PipeData.create_train_set(train_set, descriptors)
    test_X = PipeData.create_test_set(test_set, descriptors)

    # calculates intensity
    intensity_model = calculate_model(intensity_X, intensity_Y, alpha)
    intensity_result = intensity_model(test_X)

    # calculates others
    rest_result = []
    for i in range(rest_Y.shape[1]):
        rest_model = calculate_model(rest_X, rest_Y[:,i], alpha)
        rest_result = np.concatenate((rest_result, rest_model(test_X)))
    rest_result = rest_result.reshape((len(rest_result)/len(intensity_result)),len(intensity_result))

    # stacks and export results
    return np.vstack((intensity_result, rest_result))

def preprocess_cross_validation(j, intensity_X, intensity_Y, intensity_dulition, k, localized):
    #print (intensity_X.shape, intensity_Y.shape, intensity_dulition.shape)
    low_limit_intensity = int((intensity_X.shape[0] * j) / k)
    high_limit_intensity = int((intensity_X.shape[0] * (j + 1)) / k)

    new_intensity_X = (intensity_X.T[:, np.concatenate((np.arange(0, low_limit_intensity), np.arange(high_limit_intensity, intensity_X.shape[0])))]).T
    new_intensity_Y = (intensity_Y[np.concatenate((np.arange(0, low_limit_intensity), np.arange(high_limit_intensity, intensity_Y.shape[0])))])

    new_intensity_test_X = (intensity_X.T[:, range(low_limit_intensity, high_limit_intensity)]).T

    if localized:
        i = low_limit_intensity
        new_intensity_dulition = []
        while (i < high_limit_intensity):
            new_intensity_dulition = np.concatenate((new_intensity_dulition, [intensity_dulition[i]])) if len(new_intensity_dulition) > 0 else [intensity_dulition[i]]
            i += 1
    else:
        i = 0
        new_intensity_dulition = []
        while (i < low_limit_intensity or (i >= high_limit_intensity and i < len(intensity_dulition))):
            new_intensity_dulition = np.concatenate((new_intensity_dulition, [intensity_dulition[i]])) if len(new_intensity_dulition) > 0 else [intensity_dulition[i]]
            i += 1
        i = high_limit_intensity
        while (i < len(intensity_dulition)):
            new_intensity_dulition = np.concatenate((new_intensity_dulition, [intensity_dulition[i]])) if len(new_intensity_dulition) > 0 else [intensity_dulition[i]]
            i += 1

    return new_intensity_X, new_intensity_Y, new_intensity_test_X, new_intensity_dulition

def custom_cross_validation(f, p, intensity_X, intensity_Y, intensity_dulition, k, intensity, localized = True):
    intensity_result_F = []
    rest_result_F = []
    for j in range(k):
        i = 0
        new_intensity_X, new_intensity_Y, new_intensity_test_X, new_intensity_dulition = preprocess_cross_validation(j, intensity_X, intensity_Y, intensity_dulition, k, localized)
        if intensity:
            if localized:
                intensity_model = calculate_model(f, new_intensity_X, new_intensity_Y, p)
            else:
                intensity_model = f(new_intensity_X, new_intensity_Y, new_intensity_dulition, intensity)
            intensity_result = intensity_model(new_intensity_test_X)
            intensity_result = np.vstack((intensity_result, np.zeros((20, len(intensity_result)))))

        # calculates others
        if not intensity:
            if not localized:
                intensity_model = f(new_intensity_X, new_intensity_Y, new_intensity_dulition, intensity)
            intensity_result = []
            for i in range(intensity_Y.shape[1]):
                if localized:
                    intensity_model = calculate_model(f, new_intensity_X, new_intensity_Y[:,i], p)
                    intensity_result = np.concatenate((intensity_result, intensity_model(new_intensity_test_X)))
                else:
                    intensity_result = np.concatenate((intensity_result, (intensity_model[i])(new_intensity_test_X)))
            intensity_result = intensity_result.reshape(20, (len(intensity_result) / 20))
            intensity_result = np.vstack((np.zeros((intensity_result.shape[1])), intensity_result))

        intensity_result_F = np.concatenate((intensity_result_F.T, intensity_result.T)).T if len(intensity_result_F) > 0 else intensity_result
    return intensity_result_F

class PFit:
    def __init__(self, learner, ps):
        self.l = learner
        self.ps = ps
    def __call__(self, intensity_X, intensity_Y, intensity_dulition, intensity = True):
        best_p = 0
        best_score = 0
        for p in self.ps:
            intensity_result_F = custom_cross_validation(self.l, p, intensity_X, intensity_Y, intensity_dulition, k_in, intensity, True)
            IO.exportResults("output", intensity_dulition, legend, intensity_result_F)
            fscore = score(legend)
            if fscore > best_score:
                best_score = fscore
                best_p = p

        print ("ALPHA =", best_p)
        print ("BEST ALPHA SCORE", best_score)

        if intensity:
            return calculate_model(self.l, intensity_X, intensity_Y, p)

        if not intensity:
            all_models = []
            for i in range(intensity_Y.shape[1]):
                all_models = np.concatenate((all_models, [calculate_model(self.l, intensity_X, intensity_Y[:,i], p)])) if len(all_models) > 0 else [calculate_model(self.l, intensity_X, intensity_Y[:,i], p)]
            return all_models
        
        

cross_validation = True 
alpha = 0.8
k_in = 3
k_out = 2
alphas = [0.8, 5]

# reading data
train_set = DataPrep.load_data_mean_indv("TrainSet-hw2.txt")
legend = DataPrep.legend("TrainSet-hw2.txt")
descriptors = Orange.data.Table("molecular_descriptors_data.txt").X
#where_are_NaNs = np.isnan(descriptors)
#descriptors[where_are_NaNs] = 0
#descriptors = descriptors[:, np.min(descriptors, axis=0) != np.max(descriptors, axis=0)]
#print (descriptors.shape)

test_set = IO.readData("predict.txt")

# writes data for scoring.py
#with open("input", "wt") as fo:
#    for el in train_set:
#        fo.write("%s\t%s\n" % (el[0], el[1]))

# only for learning on data
intensity_X, intensity_Y, intensity_dulition ,rest_X, rest_Y, rest_dulition = PipeData.create_train_set(train_set, descriptors)

#lassoPreprocess = PFit(Orange.regression.linear.LassoRegressionLearner, alphas)
#lassoPreprocess(intensity_X, intensity_Y, intensity_dulition)
#lassoPreprocess(rest_X, rest_Y, rest_dulition, False)
#print (score(legend))
    
# -------------------------- OUTER CROSS VALIDATION
def lasso(parameter):
    return Orange.regression.linear.LassoRegressionLearner(alpha = parameter)

lassoPreprocess = PFit(lasso, alphas)

print ("INTENSITY")
int_res = custom_cross_validation(lassoPreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

print ("REST")
res_res = custom_cross_validation(lassoPreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)
print ("Over!")
