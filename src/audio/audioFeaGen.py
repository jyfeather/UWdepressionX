#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:06:06 2017

audio feature generator

@author: Yan Jin
"""

import pandas as pd
import numpy as np
import scipy.stats

'''
function definition
'''
def a_mean(l):
    return np.mean(l)

def a_min(l):
    return min(l)

def a_skewness(l):
    return scipy.stats.skew(l)

def a_kurtosis(l):
    return scipy.stats.kurtosis(l)

def a_sd(l):
    return np.std(l)

def a_median(l):
    return np.median(l)

def a_rms(l):
    # root mean square value
    return np.sqrt(np.mean(np.square(l)))

def a_peak_rms(l):
    return (max(l)/np.sqrt(np.mean(np.square(l))))

def a_iqr(l):
    # interquantile range
    return scipy.stats.iqr(l)

def a_spectral(l):
    return (scipy.stats.gmean(l)/np.mean(l))

'''
parameters
'''
dirpath = '/Users/mac/Downloads/avec2017/'
filepath = '300_P/300_COVAREP.csv'

# read audio data
df_orig = pd.read_csv(''.join([dirpath, filepath]), header=None)
df_orig.head()

fea_list = []

for col in df_orig:
    l = df_orig[col].tolist()
    if col == 0:
        l_sum = sum(l)
        l_norm = [float(i)/l_sum for i in l]
        fea_list += [a_mean(l_norm), a_min(l_norm), a_skewness(l_norm), a_kurtosis(l_norm), \
                        a_sd(l_norm), a_median(l_norm), a_rms(l_norm), a_peak_rms(l_norm), \
                        a_iqr(l_norm), a_spectral(l_norm)]
    elif col < 36:
        fea_list += [a_mean(l), a_sd(l), a_peak_rms(l), a_rms(l), a_iqr(l), a_spectral(l)]
    elif col < 74:
        fea_list += [a_mean(l), a_sd(l), a_rms(l), a_iqr(l)]