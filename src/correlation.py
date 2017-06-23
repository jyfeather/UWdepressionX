#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:10:49 2017

Correlation Analysis

@author: Yan Jin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from beeswarm import *

'''
 audio only, PHQ8_Score
'''
pd_audio = pd.read_csv('/Users/mac/Downloads/avec2017/audio_fea_train_score.csv', \
                       index_col=0)
pd_audio.shape

no_important = [989, 126, 800, 1005, 198, 270, 257, 109, 760, 240, 1192]
pd_important = pd_audio.ix[:,no_important]

corrmap = pd_important.corr()

f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmap, square=True)

'''
 audio only, PHQ8_Binary
'''
pd_audio = pd.read_csv('/Users/mac/Downloads/avec2017/audio_fea_train_binary.csv', \
                       index_col=0)
pd_audio.shape

no_important = [867, 852, 795, 307, 196, 195, 206, 362, 419, 1009, 1192]
pd_important = pd_audio.ix[:, no_important]

no = 9
bs, ax = beeswarm([pd_important[pd_important.y==0].iloc[:,no].values, \
                   pd_important[pd_important.y==1].iloc[:,no].values], \
                method = 'swarm', labels = ['0', '1'], col = ['blue', 'red'])

'''
 text only, PHQ8_Score
'''
pd_text = pd.read_csv('/Users/mac/Downloads/avec2017/text_fea_train.csv', \
                       index_col=0)
pd_text.shape

no_important = [0,1,2,3,4,5,6,7,8,10]
pd_important = pd_text.ix[:,no_important]

corrmap = pd_important.corr()

f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmap, square=True)


'''
 text only, PHQ8_Binary
'''
no = 9
bs, ax = beeswarm([pd_text[pd_text.score==0].iloc[:,no].values, \
                   pd_text[pd_text.score==1].iloc[:,no].values], \
                method = 'swarm', labels = ['0', '1'], col = ['blue', 'red'])
