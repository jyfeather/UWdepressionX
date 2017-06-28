#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:01:29 2017

Prediction

@author: Yan Jin
"""

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

h2o.init()
#h2o.ls()

'''
load data files
'''
train_audio = h2o.import_file(path = '/Users/mac/Downloads/avec2017/audio_fea_train_score.csv',\
                              header = 1)
#train_audio.describe()
dev_audio = h2o.import_file(path = '/Users/mac/Downloads/avec2017/audio_fea_dev_score.csv',\
                            header = 1)

#train_audio[[1,2,3]].head()

'''
preprocessing before running models
'''
# train, test split
train_audio, valid_audio = train_audio.split_frame(ratios=[0.75], seed=1)
#train_audio.shape
no_x_audio = len(train_audio.columns)
x_audio = train_audio.columns[:no_x_audio-1]
y_audio = train_audio.columns[no_x_audio-1]

'''
modelling
'''
# model initialization
rf_audio = H2ORandomForestEstimator(seed=12, ntrees=50, max_depth= 20, \
                                       balance_classes=False, nfolds = 5, \
                                       stopping_metric = 'MSE')

gbm_audio = H2OGradientBoostingEstimator(ntrees = 50, max_depth = 20, \
                                         distribution = 'AUTO', nfolds = 5, \
                                         stopping_metric = 'MSE')

# model training
model_audio = gbm_audio
model_audio.train(x=x_audio, y=y_audio, training_frame=train_audio, validation_frame=valid_audio)

#model_audio.show()

'''
performance checking
'''
dev_pred = model_audio.predict(dev_audio)
dev_perf = model_audio.model_performance(dev_audio)
dev_perf.show()
