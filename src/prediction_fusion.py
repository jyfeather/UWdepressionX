#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:22:28 2017

predicution by fusion

@author: Yan Jin
"""

#%reset

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score

'''
    presets
'''
isClassification = 0

'''
    function for PHQ8_binary
    
    probability based fusion
'''

'''
    function for PHQ8_score
    
    standard deviation based fusion
'''
result_audio = pd.read_csv('/Users/mac/Downloads/avec2017/tree_audio_regression.csv')
result_text = pd.read_csv('/Users/mac/Downloads/avec2017/tree_text_regression.csv')

sd_audio = list()
for i in range(1, len(result_audio.columns)):
    sd_audio.append(np.std(result_audio[[i]]))
    
sd_text = list()
for i in range(1, len(result_text.columns)):
    sd_text.append(np.std(result_text[[i]]))

result_fusion = list()
for i in range(1, len(result_audio.columns)):
    if float(sd_audio[i-1]) <= float(sd_text[i-1]):
        result_fusion.append(float(np.mean(result_audio[[i]])))
    else:
        result_fusion.append(float(np.mean(result_text[[i]])))
    
'''
    performance
'''
result_dev = pd.read_csv('/Users/mac/Downloads/avec2017/audio_fea_dev.csv', header = 0)
if isClassification == 1:
    y = result_dev.binary
else:
    y = result_dev.score


if isClassification == 1:
    f1 = f1_score(dev.binary, dev_pred)
    print(f1)
else:
    rmse = np.sqrt(np.mean((result_fusion - y)**2))
    mae = mean_absolute_error(result_fusion, y)
    print(rmse, mae)