# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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
    def exportResults2(readFileName, writeFileName, legend, results):
        results = results.clip(min=0)
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

        if intensity:
            return calculate_model(self.l, intensity_X, intensity_Y, p)

        if not intensity:
            all_models = []
            for i in range(intensity_Y.shape[1]):
                all_models = np.concatenate((all_models, [calculate_model(self.l, intensity_X, intensity_Y[:,i], p)])) if len(all_models) > 0 else [calculate_model(self.l, intensity_X, intensity_Y[:,i], p)]
            return all_models
        
        

cross_validation = True 
alpha = 0.8
k_in = 2
k_out = 10
alphas = [5]

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

print (" ----------------------- LASSO -------------------")
lasso_int_res = custom_cross_validation(lassoPreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, lasso_int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

lasso_res_res = custom_cross_validation(lassoPreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, lasso_res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)

# <codecell>

alphas = [20000]

def ridge(parameter):
    return Orange.regression.linear.RidgeRegressionLearner(alpha = parameter)

ridgePreprocess = PFit(ridge, alphas)

print (" ------------------------ RIDGE --------------------- ")

ridge_int_res = custom_cross_validation(ridgePreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, ridge_int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

ridge_res_res = custom_cross_validation(ridgePreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, ridge_res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)

# <codecell>

alphas = [1]

def elastic(parameter):
    return Orange.regression.linear.ElasticNetLearner(alpha = parameter)

elasticPreprocess = PFit(elastic, alphas)

print (" ------------------------ ELASTIC --------------------- ")

elastic_int_res = custom_cross_validation(elasticPreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, elastic_int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

elastic_res_res = custom_cross_validation(elasticPreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, elastic_res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)
print ("Over!")

# <codecell>

alphas = [100]

def random_forest(parameter):
    return Orange.classification.SimpleRandomForestLearner()

forestPreprocess = PFit(random_forest, alphas)

print (" ------------------- RANDOM FOREST ------------------- ")

forest_int_res = custom_cross_validation(forestPreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, forest_int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

forest_res_res = custom_cross_validation(forestPreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, forest_res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)

# <codecell>

alphas = [5]

def knn(parameter):
    return Orange.regression.KNNRegressionLearner()

knnPreprocess = PFit(knn, alphas)

print (" ------------------- KNN LEARNER ------------------- ")

knn_int_res = custom_cross_validation(knnPreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, knn_int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

knn_res_res = custom_cross_validation(knnPreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, knn_res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)

# <codecell>

alphas = [2000]

def linear_regression(parameter):
    return Orange.regression.linear.LinearRegressionLearner()

linearPreprocess = PFit(linear_regression, alphas)

print (" ------------------- LINEAR REGRESSION ------------------- ")

lin_int_res = custom_cross_validation(linearPreprocess, alphas, intensity_X, intensity_Y, intensity_dulition, k_out, True, False)
IO.exportResults("output", intensity_dulition, legend, lin_int_res)
fscore = score(legend)
print ("FINAL INTENSITY: ")
print (fscore)

lin_res_res = custom_cross_validation(linearPreprocess, alphas, rest_X, rest_Y, rest_dulition, k_out, False, False)
IO.exportResults("output", rest_dulition, legend, lin_res_res)
fscore = score(legend)
print ("FINAL REST: ")
print (fscore)

# <codecell>

# ----------------- MEAN -------------------------
print (" ------------------- MEAN ------------------- ")
mean_int_res = (lasso_int_res + ridge_int_res + elastic_int_res + forest_int_res + knn_int_res + lin_int_res) / 6
IO.exportResults("output", intensity_dulition, legend, mean_int_res)
fscore = score(legend)
print ("MEAN INTENSITY: ")
print (fscore)

mean_int_res = (lasso_res_res + ridge_res_res + elastic_res_res + forest_res_res + knn_res_res + lin_res_res) / 6
IO.exportResults("output", intensity_dulition, legend, mean_int_res)
fscore = score(legend)
print ("MEAN REST: ")
print (fscore)


# <codecell>

# ---------------- STACKING -------------------
print (" ----------------- STACKING ----------------- ")
intensity_matrix = np.vstack((lasso_int_res[0, :], ridge_int_res[0, :], elastic_int_res[0, :], forest_int_res[0, :], knn_int_res[0, :], lin_int_res[0, :]))

ridge_stack = Orange.regression.linear.RidgeRegressionLearner()
data = Orange.data.Table(intensity_matrix.T, intensity_Y)
intensity_model = ridge_stack(data)


all_models = []
for i in range(1,21):
    rest_matrix = np.vstack((lasso_res_res[i, :], ridge_res_res[i, :], elastic_res_res[i, :], forest_res_res[i, :], knn_res_res[i, :], lin_res_res[i, :]))
    
    data = Orange.data.Table(rest_matrix.T, rest_Y[:,i-1])
    model = ridge_stack(data)
    all_models = np.concatenate((all_models, [model])) if len(all_models) > 0 else [model]


# <codecell>

def concatenate_array(descriptors, key, value_X, value_Y=[], value=[], train=False):
    res = PipeData.indexes_array(descriptors, key)
    if (not(np.isnan(res).any())):
        value_X = np.concatenate((value_X, res))
        value_Y = np.concatenate((value_Y, value))
    else:
        if (not(train)):
            value_X = np.concatenate((value_X, np.zeros(len(res))))
    return value_X, value_Y

def create_test_set(test_set, descriptors):
    test_X = []
    i = 0
    for key in test_set:
        i += 1
        test_X, nothing = concatenate_array(descriptors, key, test_X)

    test_X = test_X.reshape((len(test_X)/descriptors.shape[1]),descriptors.shape[1])
    return test_X


# --------------------------------- GET ALL TEST DATA ----------------------------------
print (rest_Y.shape[1])
print (test_X.shape)
test_X = create_test_set(test_set, descriptors)

def calculate_algorithm_intensity(f, p):
    lass = f(p)
    data = Orange.data.Table(intensity_X, intensity_Y)
    model = lass(data)
    return model(test_X)

def calculate_algorithm_rest(f, p):
    lass = f(p)
    
    rest_result = []
    for i in range(rest_Y.shape[1]):
        data = Orange.data.Table(intensity_X, intensity_Y)
        model = lass(data)
        rest_result = np.concatenate((rest_result, model(test_X)))
    rest_result = rest_result.reshape((len(rest_result)/rest_Y.shape[1]),rest_Y.shape[1])
    return rest_result

las_int = calculate_algorithm_intensity(lasso, 5)
las_res = calculate_algorithm_rest(lasso, 5)


#intensity_result = intensity_model(test_X)
#for i in range(all_models.shape[0]):
#    rest_result = np.concatenate((rest_result, all_models[i](test_X)))
#rest_result = rest_result.reshape((len(rest_result)/len(intensity_result)),len(intensity_result))
#np.vstack((intensity_result, rest_result))

# <codecell>

rid_int = calculate_algorithm_intensity(ridge, 20000)
rid_res = calculate_algorithm_rest(ridge, 20000)

ela_int = calculate_algorithm_intensity(elastic, 1)
ela_res = calculate_algorithm_rest(elastic, 1)

for_int = calculate_algorithm_intensity(random_forest, 100)
for_res = calculate_algorithm_rest(random_forest, 100)

knn_int = calculate_algorithm_intensity(knn, 5)
knn_res = calculate_algorithm_rest(knn, 5)

lin_int = calculate_algorithm_intensity(linear_regression, 2000)
lin_res = calculate_algorithm_rest(linear_regression, 2000)

# <codecell>

def exportResults2(readFileName, writeFileName, legend, results):
    results = results.clip(min=0)
    with open(readFileName) as f:
        with open(writeFileName, "wt") as fo:
            i = 0
            for l in f:
                cid, dilution = l.strip().split('\t')
                for j in range(results.shape[0]):
                    fo.write("%s\t%s\t%f\n" % (cid, legend[j], results[j,i]))
                i += 1

intensity_test = np.vstack((las_int, rid_int, ela_int, for_int, knn_int, lin_int))
intensity_result = intensity_model(intensity_test.T)

rest_finnal = []
for i in range(las_res.shape[1]):
    rest_test = np.vstack((las_res.T[i, :], rid_res.T[i, :], ela_res.T[i, :], for_res.T[i, :], knn_res.T[i, :], lin_res.T[i, :]))
    rest_column = (all_models[i])(rest_test.T)
    rest_finnal = np.concatenate((rest_finnal, [rest_column])) if len(rest_finnal) > 0 else [rest_column]
   
finnal = np.vstack((intensity_result, rest_finnal))

exportResults2("predict.txt", "result", legend, finnal)
fscore = score(legend)

