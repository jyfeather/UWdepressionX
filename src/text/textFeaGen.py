#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:10:10 2017

Text Feature Generation

@author: Yan Jin
"""

import pandas as pd
import re

'''
functions definitions
'''
def laughter(w):
    '''
    detect word is laughter
    '''
    p = re.search('laughter', w)    
    if p is not None:
        return 1
    else:
        return 0
    
'''
parameters
'''
dirpath = '/Users/mac/Downloads/avec2017/'
filepath = '300_P/300_TRANSCRIPT.csv'

depwordpath = '/Users/mac/Documents/github/UWdepressionX/src/text/depressionWords.txt'

# read file
df_orig = pd.read_csv(''.join([dirpath, filepath]), sep = '\t')

# get data of participant
df_part = df_orig.loc[df_orig.speaker=='Participant']

num_sentence = df_part.shape[0]

# calculate the video duration for participant
duration = sum(df_part.stop_time - df_part.start_time)

# transform sentences to doc
doc_text = ' '.join(df_part['value'].tolist())

# convert it to a list of words
wordlist = doc_text.split(' ')
num_word = len(wordlist)

num_laughter = sum([laughter(w) for w in wordlist])

fea_word_ratio = num_word / duration
fea_sent_ratio = num_sentence / duration
fea_laug_ratio = num_laughter / num_word

# depression realated words
with open(depwordpath, 'r') as depf:
    dep = depf.readline()
dep_set = set([w.lower() for w in dep.strip().split(',')])

word_set = set([w.lower() for w in wordlist])

fea_dep_ratio = len(word_set & dep_set) / num_word
