import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from tree_helpers import SkNode, create_sk_nodes
from sklearn.metrics import precision_score, recall_score

def get_errs(model, features, data, bor, kfold):

    errs = np.array([])
    indices = np.array([])
    

    
    if bor is not None:
        for train_index, test_index in kfold.split(data, data.borough):
            train_set = data.iloc[train_index]
            test_set = data.iloc[test_index]
            model.fit(train_set[features], train_set.rating)

            test_set_bor = test_set[test_set.borough == bor]
            bor_test = test_set_bor.index.to_numpy()
            preds = model.predict(test_set_bor[features])
    
            diff = test_set_bor.rating - preds
            errs = np.concatenate((errs, np.abs(diff)))
            indices = np.concatenate((indices, bor_test))
        return errs, indices
    else:
        for train_index, test_index in kfold.split(data):
            train_set = data.iloc[train_index]
            test_set = data.iloc[test_index]
            model.fit(train_set[features], train_set.rating)
            preds = model.predict(test_set[features])
            index = test_set.index.to_numpy()
    
            diff = test_set.rating - preds
            errs = np.concatenate((errs, np.abs(diff)))
            indices = np.concatenate((indices, index))
        return errs, indices
    
def assign_class(data, bor, errs, indices, thresh):
    if bor is not None:
        data = data[data.borough == bor]
    "assign error values to borough-specific dataframe"
    data.loc[indices, 'Errors'] = errs 

    "classify large errors"
    data['Class'] = data['Errors'].map(lambda x: x > thresh).astype(int)
    return data




class LE_Class():
    def __init__(self, class_features):
        self.features = class_features
        l = len(class_features)
        self.feat_imp = pd.DataFrame({'features': class_features, 'importance score': l*[0]})
        self.tree = None
        self.prec = 0
        self.recall = 0

    def fit(self, X):
        self.tree = DecisionTreeClassifier(
            min_samples_leaf = 5,  
            random_state= 216)

        self.tree.fit(X[self.features], X['Class'])
        preds = self.tree.predict(X[self.features])
        self.prec = np.round(precision_score(X['Class'], preds), 4)
        self.recall = np.round(recall_score(X['Class'], preds), 4)

        self.feat_imp['importance score'] =  self.tree.feature_importances_
        self.feat_imp.sort_values('importance score', ascending = False, inplace = True)
        
def get_subsets(nodes):
    "collect the coordinate constraints implicated for each leaf node"
    endpts = {}
    i=0
    for node in nodes.values():
        if (node.is_leaf) and (node.prediction == 1):
            endpts[i] = node.get_constraints()
            i += 1
    num_subsets = len(endpts)

    "rearrange to the corresponding Cartesian product of coordinate sets for each leaf"
    subsets = {key: [] for key in range(num_subsets)}
    i = 0
    for coord_bds in endpts.values():
        for j in range(len(coord_bds[0])):
            list = [coord_bds[0][j], coord_bds[1][j]]
            subsets[i].append(list)
        i += 1
    return subsets

"locate the cluster"
def def_cluster(data, subsets, class_features):
    cluster = {}
    feature_num = len(class_features)
    num_subsets = len(subsets)

    for i in range(num_subsets):
        cluster[i] = data
        for j in range(feature_num):
            feature = class_features[j]
            cluster[i] = cluster[i].loc[(cluster[i][feature] > subsets[i][j][0]) &
                                 (cluster[i][feature] < subsets[i][j][1])]

    whole = pd.DataFrame({})
    for i in range(num_subsets):
        whole = pd.concat([whole, cluster[i]])
    return whole

def deconstruct(model, mod_feats, class_feats, data, bor, kfold, thresh):
    list = []
    errs, index = get_errs(model, mod_feats, data, bor, kfold)
    data = assign_class(data, bor, errs, index, thresh)
    le = LE_Class(class_feats)
    le.fit(data)
    list.append(le)
    nodes = create_sk_nodes(le.tree)

    subsets = get_subsets(nodes)
    list.append(subsets)
    err_df = def_cluster(data, subsets, class_feats)
    list.append(err_df)
    list.append(good_rmse(err_df, data))
    list.append(bad_rmse(err_df))
    return list





def good_rmse(data, amb_data):

    index_to_remove = set(data.index.tolist())
    index = set(amb_data.index.tolist())
    index = index.difference(index_to_remove)
    index = [*index]


    squares = amb_data.loc[index, 'Errors']**2
    return np.sqrt(squares.mean())



def bad_rmse(data):

    squares = data['Errors']**2
    return np.sqrt(squares.mean())



class DecCluster():

    """
    DecCluster is a form of greedy algorithm aimed at applying a best initial model to a ratings-based dataframe D and pinpointing
    a feature-based description of the data subset S where the model behaves horribly, i.e. returns predictions of ratings with severely large error. 
    It is a class that takes a 'KFold' or 'StratifiedKFold' object as input, and can be fitted to any dataframe D with 'borough' and 'ratings'
    columns with a (unfitted) model selected that ideally is the best initial approximation to the ratings within D. When fitting a DecCLuster object, the features
    used for the model must be inputted as well, since the model is fitted in due course, and the features to be used for classifying 
    S must also be inputted, along with a threshold value that limits the lowest error desired in S.

    """
    def __init__(self,  kfold):
        self.data = None
        self.bor = None
        self.model = None
        self.kfold = kfold
        self.mod_feats = None
        self.class_feats = None
        self.thresh = None
        self.err_tree = None
        
        self.feat_imp = None
        self.prec = 0
        self.recall = 0
        "a dictionary of lists, each corresponding to a different Cartesian product"
        self.subsets = None 
        "the 'bad' data subset, or the set with very large errors under self.model"
        self.err_df = None

        self.good_rmse = None
        self.bad_rmse = None

    def fit(self, data, bor, model, mod_feats, class_feats, thresh):
        self.data = data
        self.bor = bor
        self.model = model
        self.mod_feats = mod_feats
        self.class_feats = class_feats
        self.thresh = thresh
        results = deconstruct(model, mod_feats, class_feats, data, bor, self.kfold, thresh)
        le = results[0]

        self.tree = le.tree
        self.prec = le.prec
        self.recall = le.recall
        self.feat_imp = le.feat_imp
        self.subsets = results[1]
        self.err_df = results[2]

        self.good_rmse = results[3]
        self.bad_rmse = results[4]



        