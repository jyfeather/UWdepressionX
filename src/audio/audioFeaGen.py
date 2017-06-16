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
import os
import csv
from scipy.fftpack import dct

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

def a_dct(l):
    return dct(l, type=2).tolist()[:10]

'''
parameters
'''
dirpath = '/Users/mac/Downloads/avec2017/'
dirlist = os.listdir(dirpath)

with open(dirpath + 'audio_fea.csv', 'w') as outfile:
    wr = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    
    for fold in dirlist:
        if not fold.endswith('P'):
            continue
        #filepath = dirpath + fold + '/' + fold[:3] + '_COVAREP.csv'
        filepath = fold + '/' + fold[:3] + '_COVAREP.csv'
        print(filepath)
        
        filepath2 = fold + '/' + fold[:3] + '_FORMANT.csv'
    
        # read audio data
        df_orig = pd.read_csv(''.join([dirpath, filepath]), header=None)
        df_orig2 = pd.read_csv(''.join([dirpath, filepath2]), header=None)
        #df_orig.head()
        
        fea_list = []
        
        for col in df_orig:
            l = df_orig[col].tolist()
            if col == 0:
                l_sum = sum(l)
                l_norm = [float(i)/l_sum for i in l]
                fea_list += [a_mean(l_norm), a_min(l_norm), a_skewness(l_norm), a_kurtosis(l_norm), \
                                a_sd(l_norm), a_median(l_norm), a_rms(l_norm), a_peak_rms(l_norm), \
                                a_iqr(l_norm), a_spectral(l_norm)]
                fea_list += a_dct(l)
            elif col < 36:
                fea_list += [a_mean(l), a_sd(l), a_peak_rms(l), a_rms(l), a_iqr(l), a_spectral(l)]
                fea_list += a_dct(l)
            elif col < 74:
                fea_list += [a_mean(l), a_sd(l), a_rms(l), a_iqr(l)]
                fea_list += a_dct(l)
        
        for col in df_orig2:
            l = df_orig[col].tolist()
            fea_list += [a_mean(l), a_sd(l), a_peak_rms(l), a_rms(l), a_iqr(l), a_spectral(l)]
            fea_list += a_dct(l)
            
        fea_list = [round(x, 3) for x in fea_list]
        
        wr.writerow(fea_list)