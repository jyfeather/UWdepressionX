#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:20:56 2017

silence detector 

@author: Yan Jin
"""

from pydub import AudioSegment
from pydub.utils import db_to_float
from functools import reduce

audio = AudioSegment.from_wav('/Users/mac/Downloads/avec2017/300_P/300_AUDIO.wav')

# the average volume of the audio
average_loudness = audio.rms

# anything that is 30 decibels quiter than the rms to be silence
silence_threshold = average_loudness * db_to_float(-30)

# filter out the silence
audio_silence = (ms for ms in audio if ms.rms > silence_threshold)

# combine all the chunks together
#audio_no_silence = reduce(lambda a, b: a+b, audio_silence)
