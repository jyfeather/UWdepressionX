#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:11:57 2017

prediction

@author: yan jin
"""

#%reset

import pandas as pd
import os
import matplotlib.pyplot as plt

# read input data
fea_audio = pd.read_csv('/Users/mac/Downloads/avec2017/audio_fea.csv', header = None)
fea_text = pd.read_csv('/Users/mac/Downloads/avec2017/text_fea.csv', header = None)

# read train/dev/test indicators
list_train = pd.read_csv('/Users/mac/Downloads/avec2017/train_split_Depression_AVEC2017.csv')
list_dev = pd.read_csv('/Users/mac/Downloads/avec2017/dev_split_Depression_AVEC2017.csv')
list_test = pd.read_csv('/Users/mac/Downloads/avec2017/test_split_Depression_AVEC2017.csv')

# list directory
dirpath = '/Users/mac/Downloads/avec2017/'
dirlist = os.listdir(dirpath)
list_fold = []
for fold in dirlist:
    if not fold.endswith('P'):
            continue
    list_fold.append(int(fold[:3]))

# get train/dev/test exactly
list_train = list_train.loc[[i in list_fold for i in list_train['Participant_ID'].tolist()],]
list_dev = list_dev.loc[[i in list_fold for i in list_dev['Participant_ID'].tolist()],]
list_test = list_test.loc[[i in list_fold for i in list_test['participant_ID'].tolist()],]

fea_audio = fea_audio.fillna(method='ffill')
fea_audio_train = fea_audio.loc[[i in list_train['Participant_ID'].tolist() for i in list_fold],]
fea_text_train = fea_text.loc[[i in list_train['Participant_ID'].tolist() for i in list_fold],]
fea_audio_dev = fea_audio.loc[[i in list_dev['Participant_ID'].tolist() for i in list_fold],]
fea_text_dev = fea_text.loc[[i in list_dev['Participant_ID'].tolist() for i in list_fold],]
fea_audio_test = fea_audio.loc[[i in list_test['participant_ID'].tolist() for i in list_fold],]
fea_text_test = fea_text.loc[[i in list_test['participant_ID'].tolist() for i in list_fold],]

# distribution illustration
#plt.hist(list_train['PHQ8_Binary'])
#plt.hist(list_dev['PHQ8_Binary'])
#plt.hist(list_train['PHQ8_Score'])
#plt.hist(list_dev['PHQ8_Score'])

# dump audio only
fea_audio_train['binary'] = pd.factorize(list_train['PHQ8_Binary'])[0]
fea_audio_dev['binary'] = pd.factorize(list_dev['PHQ8_Binary'])[0]
fea_audio_train['score'] = pd.factorize(list_train['PHQ8_Score'])[0]
fea_audio_dev['score'] = pd.factorize(list_dev['PHQ8_Score'])[0]
fea_audio_train['gender'] = pd.factorize(list_train['Gender'])[0]
fea_audio_dev['gender'] = pd.factorize(list_dev['Gender'])[0]
fea_audio_train.to_csv('/Users/mac/Downloads/avec2017/audio_fea_train.csv')
fea_audio_dev.to_csv('/Users/mac/Downloads/avec2017/audio_fea_dev.csv')

# dump binary & score, text only
fea_text_train['binary'] = pd.factorize(list_train['PHQ8_Binary'])[0]
fea_text_train['score'] = pd.factorize(list_train['PHQ8_Score'])[0]
fea_text_dev['binary'] = pd.factorize(list_dev['PHQ8_Binary'])[0]
fea_text_dev['score'] = pd.factorize(list_dev['PHQ8_Score'])[0]
fea_text_train['gender'] = pd.factorize(list_train['Gender'])[0]
fea_text_dev['gender'] = pd.factorize(list_dev['Gender'])[0]
fea_text_train.to_csv('/Users/mac/Downloads/avec2017/text_fea_train.csv')
fea_text_dev.to_csv('/Users/mac/Downloads/avec2017/text_fea_dev.csv')

"""
    predictive modeling
"""
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import f1_score
#
#y = pd.factorize(list_train['PHQ8_Binary'])[0]
#
#clf = RandomForestClassifier()
##clf.fit(fea_text_train, y).score(fea_text_train, y)
#clf.fit(fea_audio_train, y)
#
#pd.crosstab(clf.predict(fea_text_dev), list_dev['PHQ8_Binary'], rownames=['pred'], colnames=['actual'])
#list(zip(fea_text_train, clf.feature_importances_))
#f1_score(clf.predict(fea_text_dev), list_dev['PHQ8_Binary'])
