#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 00:15:19 2018

@author: abhijay
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import matplotlib
from bayes_opt import BayesianOptimization

font = {'size'   : 14}
matplotlib.rc('font', **font)

import os
os.chdir('/home/abhijay/Documents/ML/hw_3/Q_7/')

def make_target_variable(data):
    data['salary-bracket'] = data['salary-bracket'].apply(lambda y: 0 if y==" <=50K" else 1)
    return data

def find_categorical_continuous_features(data):
    categorical_features = [data.columns[col] for col, col_type in enumerate(data.dtypes) if col_type == np.dtype('O') ]
    continuous_features = list(set(data.columns) - set(categorical_features))
    return categorical_features, continuous_features

def plot_results(dev_accuracy, x_label, hyperparameter_values, saveAs):
    w = 0.5
    ind = np.arange(len(dev_accuracy))
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(ind, dev_accuracy, width=w, label = 'Validation Accuracy')
    ax.set_ylabel( 'Accuracy', fontsize=15)
    ax.set_xlabel( x_label, fontsize=15)
    ax.set_xticks(ind+w)
    ax.set_xticklabels([str(ind[i]+1)+"- ("+str(int(round(val[0])))+","+str(int(round(val[1],2)))+")" for i,val in enumerate(hyperparameter_values)])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.set_title(saveAs)
    plt.xticks(rotation=70)
    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig(saveAs)

def bagging_classifier(maxDepthOfTree, noOfTrees):
    
    clf = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=int(round(maxDepthOfTree,0))),n_estimators=int(round(noOfTrees,0)))
    
    clf.fit(x, y)
    
    return clf.score(x_dev, y_dev)

def boosting_classifier(maxDepthOfTree, noOfTrees):
    
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=int(round(maxDepthOfTree,0))),n_estimators=int(round(noOfTrees,0)))
    
    clf.fit(x, y)
    
    return clf.score(x_dev, y_dev)

if __name__ == "__main__":
    
    col_names = ["age","workclass","education","marital-status","occupation","race","gender","hours-per-week","native-country","salary-bracket"]
    
    ##### Load data #####
    train_data = pd.read_csv("income-data/income.train.txt", names = col_names)
    dev_data = pd.read_csv("income-data/income.dev.txt", names = col_names)
    test_data = pd.read_csv("income-data/income.test.txt", names = col_names)
    
    train_data = make_target_variable(train_data)
    test_data = make_target_variable(test_data)
    dev_data = make_target_variable(dev_data)
        
    categorical_features_, continuous_features_ = find_categorical_continuous_features(train_data.iloc[:,0:-1])
    
    categorical_features = [train_data.columns.get_loc(c) for c in categorical_features_]
    
    continuous_features = [train_data.columns.get_loc(c) for c in continuous_features_]
    
    ##### Encoding categorical values to labels #####
    le = preprocessing.LabelEncoder()
    all_df = pd.concat([train_data,test_data,dev_data])
    for feature in categorical_features_:
        le.fit(all_df[feature])
        train_data[feature] = le.transform(train_data[feature])
        test_data[feature] = le.transform(test_data[feature])
        dev_data[feature] = le.transform(dev_data[feature])
    
    featuresUniqueValues = [train_data[col].unique() for col in col_names]
    
    ##### Convert pandas dataframe to numpy array #####
    x = train_data.iloc[:,0:train_data.shape[1]-1].values
    y = (train_data.values)[:,-1]
    
    x_test = test_data.iloc[:,0:test_data.shape[1]-1].values
    y_test = (test_data.values)[:,-1]
    
    x_dev = dev_data.iloc[:,0:dev_data.shape[1]-1].values
    y_dev = (dev_data.values)[:,-1]
    
    param_dict = {
     'maxDepthOfTree': (1, 20),
     'noOfTrees':(1,100)
    }
    
    BO = BayesianOptimization(bagging_classifier,param_dict)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        BO.maximize(init_points=1, n_iter=50, acq='ei', xi=0.0)
    
    x_label = "Iteration - (Max depth of tree, No. of trees)"
    saveAs = 'Bayesian Optimization on Bagging Classifier'
    plot_results(BO.Y*100, x_label, BO.X, saveAs)
    
    BO = BayesianOptimization(boosting_classifier,param_dict)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        BO.maximize(init_points=1, n_iter=50, acq='ei', xi=0.0)
    
    x_label = "Iteration - (Max depth of tree, No. of trees)"
    saveAs = 'Bayesian Optimization on Boosting Classifier'
    plot_results(BO.Y*100, x_label, BO.X, saveAs)
    