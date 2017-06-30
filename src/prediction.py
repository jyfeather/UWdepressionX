#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:01:29 2017

Prediction with scikit-learn

@author: Yan Jin
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

'''
preset
'''
isClassification = 1

'''
load data files
'''
train = pd.read_csv('/Users/mac/Downloads/avec2017/text_fea_train.csv', header = 0)
#train_audio.describe()
dev = pd.read_csv('/Users/mac/Downloads/avec2017/text_fea_dev.csv', header = 0)

#train_audio[[1,2,3]].head()

'''
preprocessing before running models
'''
# train, test split
train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
train, valid = train[train['is_train']==True], train[train['is_train']==False]

#train_audio.shape
no_x = len(train.columns)
x = train.columns[1:no_x-3]
if isClassification == 1:
    y = train.columns[no_x-3]
else:
    y = train.columns[no_x-2]

'''
modelling
'''
# model initialization
if isClassification == 1:
    rf = RandomForestClassifier(n_estimators = 50, n_jobs = -1)
else:
    rf = RandomForestRegressor(n_estimators = 50, criterion = 'mse', n_jobs = -1)

# model training
if isClassification == 1:
    rf.fit(train[x], train[y].astype('category'))
else:
    rf.fit(train[x], train[y])

#model_audio.show()

'''
performance checking
'''
dev_pred = rf.predict(dev[x])
#dev_pred2 = model_audio.predict_leaf_node_assignment(dev_audio)
#h2o.download_csv(dev_pred, '/Users/mac/Downloads/test.csv')

if isClassification == 1:
    f1 = f1_score(dev.binary, dev_pred)
else:
    rmse = np.sqrt(np.mean((dev.score - dev_pred)**2))
    mae = mean_absolute_error(dev.score, dev_pred)

'''
each tree prediction
'''
per_tree_pred = [tree.predict(dev[x]) for tree in rf.estimators_]

per_tree_df = pd.DataFrame(per_tree_pred)
per_tree_sd = list()
for i in range(len(per_tree_df.columns)):
    per_tree_sd.append(np.std(per_tree_df[[i]]))

