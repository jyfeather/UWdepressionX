#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:43:30 2017

MFCC

wav.read does not work for our audio file

@author: Yan Jin
"""

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

fold_dir = '/Users/mac/Downloads/avec2017/'

(rate,sig) = wav.read(''.join([fold_dir, '301_P/301_AUDIO.wav']))
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])
