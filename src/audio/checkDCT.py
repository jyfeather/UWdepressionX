#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:41:08 2017

DCT validatio

@author: Yan Jin
"""

import pandas as pd
import numpy as np
import scipy.stats
import os
import csv
from scipy.fftpack import dct, idct
from numpy import linalg
import matplotlib.pyplot as plt

def a_dct(l, m):
    tmp = dct(l, type=2, norm = 'ortho')
    tmp_idx = sorted(range(len(tmp)), key=lambda k: -abs(tmp[k]))
    tmp[tmp_idx[m:]] = 0
    return tmp

'''
parameters
'''
# read audio data
df_orig = pd.read_csv('/Users/mac/Downloads/avec2017/300_P/300_COVAREP.csv', header=None)
df_orig2 = pd.read_csv('/Users/mac/Downloads/avec2017/300_P/300_FORMANT.csv', header=None)
#df_orig.head()

col = 20
l = df_orig[col].tolist()

dist_list = list()
for m in range(300, 310, 1):    
    l_idct = a_dct(l, m)
    
    l2 = idct(l_idct, type = 2, norm = 'ortho')
    
    dist_list.append(linalg.norm(l-l2))

plt.plot(dist_list)
