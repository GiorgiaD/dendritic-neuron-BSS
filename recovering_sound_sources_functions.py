# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:33:37 2019

@author: Giorgia
"""
# import libraries
# for dendritic neurons
from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.matlib
import os
import shutil
import sys
import matplotlib.cm as cm
# for audio
#from pydub import AudioSegment
#from pydub.playback import play
#from pydub import effects
#import librosa
#import librosa.display
import numpy as np
import math
import matplotlib.pyplot as plt

# activation function
beta = 5
def g(x):
    alpha = 1
    theta= 0.5
    ans = 1/(1+alpha*np.exp(beta*(-x+theta)))
    return ans

# functions for audio
def next_power_of_2(x):
    """find a power of 2 """
    assert x >= 0
    if x == 0:
        return 1
    else:
        return 2 ** math.ceil(math.log2(x))
    
    # functions for audio
def extract_mel(filename,sample_rate,n_fft,hop_length,n_mels,plot_fig):
    data, sr = librosa.load(filename,sr=sample_rate)
    eps = np.finfo(np.float32).eps
    
    """
    # option 1 (more fuzzy)
    stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    mel = np.log(eps+librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length))
    """
    # option 2 (more neat)
    stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    mel = (librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length))
    mel = librosa.power_to_db(mel**2, ref=np.max)
    """
    which is the equivalent of:

    ps = librosa.feature.melspectrogram(y=data, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels)
    ps_db= librosa.power_to_db(ps, ref=np.max)

    plt.figure(figsize=(6,3))
    librosa.display.specshow(ps_db, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title('Mel frequency spectrogram')

    """
    if plot_fig:
        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        librosa.display.specshow(mel, x_axis='time', y_axis='mel')
        #print(mel.shape)
        plt.colorbar()
        plt.title('Mel frequency spectrogram for '+filename)
        
        plt.subplot(1,2,2)
        librosa.display.waveplot(data, sr=sr)
    return data,mel

# functions for audio
def combine_tracks(file1,file2,idx):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)
    combined = sound1.overlay(sound2)
    combined = effects.normalize(combined)
    filename_combined = "combined"+str(idx)+".wav"
    combined.export(filename_combined, format='wav')