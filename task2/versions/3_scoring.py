import scipy.stats
import numpy as np
import math
from collections import defaultdict
import random

def pearson(x,y):
    x,y = np.array(x), np.array(y)
    anynan = np.logical_or(np.isnan(x), np.isnan(y))
    r = scipy.stats.pearsonr(x[~anynan],y[~anynan])[0]
    return 0. if math.isnan(r) else r

def fix_dilution(s):
    return s.replace('"', '').strip()

def legend(fn):
    with open(fn) as f:
        for l in f:
            vals_legend = l.strip().split('\t')[6:]
            return vals_legend
 
def load_data(fn):
    readings = defaultdict(list)
    with open(fn) as f:
        vals_legend = next(f).strip().split('\t')[6:]
        for l in f:
            l = l.strip()
            t = l.split('\t')
            cid, dilution, vals = t[0], fix_dilution(t[4]), list(map(float, t[6:]))
            readings[cid, dilution].append(vals)
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
    readings = load_data(fn)
    r2 = {}
    for a,b in readings.items():
        r2[a] = mean_indv_notnan(np.array(b))
    return r2

def permuted_chem(readings, perm):
    rand = random.Random(perm)
    comb = sorted(readings.items())
    keys = [ a[0] for a in comb ]
    values = [ a[1] for a in comb ]
    rand.shuffle(values)
    return { a:b for a,b in zip(keys, values) }


def read_predictions(s):
    rd = {}
    leg = set(LEGEND)
    for l in s.split("\n"):
        l = l.strip()
        if l:
            cid, desc, val = t = l.split('\t')
            if desc not in leg:
                raise ValueError("Descriptor " + desc + " not valid.")
            rd[cid, desc] = float(val)
    return rd

def preds_to_vec(pd):
    rd = {}
    cids = [a[0] for a in pd.keys()]
    for cid in set(cids):
        rd[cid] = np.array([ pd[cid, desc] for desc in LEGEND ])
    return rd
 
NAN = float("NaN")

def realscore(a, answers):
    #intensity has a fixed dilution
    return list(answers.get((a[0], "1/1,000"), [NAN])[:1]) + list(answers[a][1:])

def evaluate_r(preds, query, answers):
    userscores = np.array([ preds[a[0]] for a in query ])
    realscores = np.array([ realscore(a, answers) for a in query ])
    rint = pearson(userscores[:,0], realscores[:,0])
    rval = pearson(userscores[:,1], realscores[:,1])
    rdecall = [ pearson(userscores[:,i], realscores[:,i]) for i in range(2,21) ]
    rdec = np.mean(rdecall)
    return np.array([rint, rval, rdec])

def read_query(fn):
    res = []
    with open(fn) as f:
        for l in f:
            l = l.strip()
            if l:
                res.append(tuple(l.split("\t")))
    return res

def normalization_consts(query, answers):
    """ Obtain the gold-standard deviation for our test and
    leaderboard data sets. """
    reals = { cid: realscore((cid, dil), answers) for cid,dil in query }
    permres = []
    for i in range(10000):
        permuted = permuted_chem(answers, i)
        permres.append(evaluate_r(reals, query, permuted))
    permres = np.array(permres)
    return np.mean(permres, axis=0), np.std(permres, axis=0)

NORM_STD = [ 0.18, 0.16, 0.06 ] #an average of normalizatin_costs outputs)
#means were 0 (as expected for Pearson correlation)

#NORM_STD = [0.12539162,  0.09908872,  0.03480594]

def final_score(rs):
    zs = rs/NORM_STD
    return np.mean(zs)

if __name__ == "__main__":

    LEGEND = legend("TrainSet-hw2.txt")

    query = read_query("input")
    train = load_data_mean_indv("TrainSet-hw2.txt")
    #print (query[0] in train)
    #print (train)

    test = load_data_mean_indv("TrainSet-hw2.txt")
#    leaderboard = load_data_mean_indv("LeaderboardSet-hw2.txt")


    query_test = [ a for a in query if a in train ]
    print (query_test)
#    query_leaderboard = [ a for a in query if a in leaderboard ]

    #print("NORM_TEST", normalization_consts(query_test, test))
    #print("NORM_LEAD", normalization_consts(query_leaderboard, leaderboard))
    #fdsfd

    preds = read_predictions(open("output").read())
    preds = preds_to_vec(preds)

    #print (preds)

    #mypreds = read_predictions(open("result").read())
    #mypreds = preds_to_vec(preds)

    rs = evaluate_r(preds, query, train)
    print("FINAL", final_score(rs))
    #rs = evaluate_r(mypreds, query, leaderboard)
    #print("LEADERBORD", final_score(rs))


