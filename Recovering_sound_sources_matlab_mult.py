# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:18:40 2019

@author: Giorgia
"""

# import libraries
# for dendritic neurons
from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import numpy.matlib
import os
import shutil
import sys
import matplotlib.cm as cm
# for audio
from pydub import AudioSegment
from pydub.playback import play
from pydub import effects
import librosa
import librosa.display
import math
import random
import matplotlib.pyplot as plt
import scipy.special
from recovering_sound_sources_functions import *
from argparse import ArgumentParser
from glob import glob
import timeit

from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score



mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
params = {'backend': 'ps',
    'axes.labelsize': 11,
    'text.fontsize': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.figsize': [10 / 2.54, 6 / 2.54]}


#%%

def parse_args():
    parser = ArgumentParser(description='Testing dendritic neuron for sound source separation')
    
    parser.add_argument(
        '-e_n', '--exp_name',
        type=str, default='exp_',
        help='Name/Number of the experiment e.g. to save files'
        )
    parser.add_argument(
        '-n_sounds', '--n_sounds',
        type=int, default='2',
        help='Number of sounds to combine the target with'
        )
    parser.add_argument(
        '-N', '--N',
        type=int, default='8',
        help='Number of output neurons'
        )
    parser.add_argument(
        '-n_tr', '--n_rep_train',
        type=int, default='1000',#3000
        help='Number of repetitions of mixture sequence in training'
        )
    parser.add_argument(
        '-n_te', '--n_rep_test',
        type=int, default='10',#50
        help='Number of repetitions of sounds sequence in testing'
        )
    parser.add_argument(
        '-l', '--loop',
        type=int, default='10',#20
        help='Number of loops during testing'
        )
    parser.add_argument(
        '-t_s', '--t_steps',
        type=int, default='400',
        help='Number of time steps for each mixture/sound presentation'
        )
    parser.add_argument(
        '-n_r', '--n_repet',
        type=int, default='10',
        help='Number of repetition for each simulation'
        )
    parser.add_argument(
        '-noise', '--noise',
        type=float, default='0.1',
        help='Strength of output noise in the learning rule.'
        )
    parser.add_argument(
        '-a_c', '--all_comb',
        action='store_true',
        help='Choose if you want all combinations of mixtures'
        )
    parser.add_argument(
        '-p_i', '--plastic_inhib',
        action='store_true',
        help='Choose if you want plastic inhibition'
        )
    parser.add_argument(
        '-r_w', '--record_w',
        action='store_true',
        help='Choose if you want to monitor the average weigth change over time'
        )
    parser.add_argument(
        '-n_inh', '--normalize_inhibition',
        action='store_true',
        help='Choose if you want to normalize the strength of inhibition wrt N'
        )
    parser.add_argument(
        '-s_c', '--sparse_connectivity',
        action='store_true',
        help='Choose if you want to have sparse connectivity (30 per cent)'
        )
    parser.add_argument(
        '-t_m', '--test_mixt',
        action='store_true',
        help='Choose if you want to test also the response to mixtures'
        )
    parser.add_argument(
        '-e', '--eps',
        type=float, default=10**(-5)*0.5,
        help='Learning rate'
        ) 
      
    return parser.parse_args()


# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    # parameters simulation
    args = parse_args()    
    exp_name = args.exp_name
    n_sounds = args.n_sounds # number of mixtures
    n_sounds_tot = 1+n_sounds
    N = args.N # number of output neurons (=number of mixed sounds plus target)
    n_rep_train = args.n_rep_train
    n_rep_test = args.n_rep_test
    loop = args.loop #outpus are averaged over 20 trials
    t_steps = args.t_steps # number of steps to be performed per time frame
    noise = args.noise #strength of output noise in the learning rule. In the manuscript, we changed this in [0,0.9].
    n_repet = args.n_repet

    path = "dataset_matlab_txt/"
    all_comb = args.all_comb # set to True is you want all possible combinations of the noises
    plastic_inhib = args.plastic_inhib
    plastic_inhib = True
    record_w = args.record_w # set to True if you want to record how much weights change
    normalize_inhibition = args.normalize_inhibition # set to True to normalize the inhibition strenght wrt N
    normalize_inhibition = True
    test_mixt = args.test_mixt # set to True if you want to test also the response to mixtures
    sparse_connectivity = args.sparse_connectivity
    if all_comb==False:
        n_mixt = n_sounds
    else:
        n_mixt = int(scipy.special.binom(n_sounds+1, 2))
    plot_distr = True

    # load the full dataset
    n_class = 5
    n_elem = 10
    dataset = []
    n_i = 0
    for n_c in range(n_class):
        for n_e in range(n_elem):
            filename = 'spectro'+str(n_i)+'.txt'
            data = np.loadtxt(path+filename)
            dataset.append(data)
            n_i += 1
        
    
    # hyperparameters for mel spectrograms
    n_mels = np.shape(dataset[0])[0]
    print("mel shape is: ",np.shape(dataset[0]))

    # training
    dt = 1
    eps = args.eps #learning rate

    t0 = n_mixt*t_steps #time intervals for calculating the mean and variance of activity.
    
    # model hyperparameters
    beta = 5
    tau =15 # membrane time constant
    tau_syn = 5 # synaptic time constant
    n_in = n_mels*np.shape(dataset[0])[1] # number of input neurons is the numbers of pixels in the spectrogram
    PSP = np.zeros(n_in) #post synaptic potentials
    I_syn = np.zeros(n_in) # synaptic current
    g_L = 1/tau #leak conductance of soma
    g_d = 0.7 #strength of dendrite to somatic interaction
    
    # initialize the list to store the log likelihood for targets and distractors
    prob_targets_all = np.array(())
    prob_distr_all = np.array(())
    auc_all = []
    
    # create a directory to save the results
    save_path = exp_name
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, exp_name)
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
    #try:
    #    os.mkdir(save_path)
    #except OSError:
    #    print ("Creation of the directory %s failed" % path)
    #else:
    #    print ("Successfully created the directory %s " % path)
    
    save_path = exp_name+'/'
    # write on files set-up and results of simulations
    filename = save_path+'res'+exp_name+'.txt'
    file = open(filename,'w')
    file.write('Results for simulation with the following hyperparameters ')
    file.write('\n Experiment name = ')
    file.write(exp_name)
    file.write('\n Number of sounds (excluding target) ')
    file.write(str(n_sounds))
    file.write('\n Number of simulations ')
    file.write(str(n_repet))
    file.write('\n Number of training epochs ')
    file.write(str(n_rep_train))
    file.write('\n Number of testing epochs ')
    file.write(str(n_rep_test))
    
    for sim_n in range(n_repet):
        print('############### SIMULATION {} #################'.format(sim_n))
        file.write('\n SIMULATION ')
        file.write(str(sim_n))
        w  = np.random.randn(n_in,N)/np.sqrt(n_in) #synaptic weights projecting to dendrite. This will be trained.
        if sparse_connectivity: # add sparse connectivity (set to zero 30% of connections)
            print('Sparse connectivity: set to zero 30% of connections')
            w_fl = w.flatten()
            tot_syn = np.size(w_fl)
            connect_perc = 0.3
            connections_zero = int(tot_syn*connect_perc)
            connections_zero_idx = np.random.choice(np.arange(tot_syn),connections_zero)
            for i in connections_zero_idx:
                w_fl[i] = 0
            w = np.reshape(w_fl,(n_in,N))
        if record_w:
            w_change_gap = 2000
            w_ave = 100
            w_lists = []
            for n_out in range(N):
                new_w_lists = np.zeros((n_in,w_ave))
                w_lists.append(new_w_lists)
            w_change = []
        
        gain = 10 #maximum firing rate of input neurons
        
        # fixed lateral inhibition
        if normalize_inhibition == False:
            w_inh_max = 0.4
        elif normalize_inhibition:
            w_inh_max = 0.4/N
        w_inh =np.ones((N,N))*w_inh_max
        w_inh[w_inh<0] = 0
        w_inh[w_inh>w_inh_max] = w_inh_max
        
        mask = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    mask[i,j] = 1
        w_inh *= mask
        # plastic lateral inhibition
        if plastic_inhib:
            cA_p = -1*0.105*0.1
            cA_d = 0.525*0.1*0.1
            tau_p=20
            tau_d=40
            A_pre_p=np.zeros(N)
            A_pre_d=np.zeros(N)
            A_post_p=np.zeros(N)
            A_post_d=np.zeros(N)
        
        # initialize values
        V_som_list=np.zeros((N,t0))
        V_dend = np.zeros(N)
        V_som = np.zeros(N)
        f = np.zeros(N) #output firing rate. Here, this takes the value in [0,1].
    
        # create my source
        # 1. randomly pick (one) target, (n_sounds) sounds for mixtures, and (one+n_sounds) sounds as distractors from the same distributions
        n_sounds_list = random.sample(list(np.arange(n_class*n_elem)), n_sounds_tot) # sample the gaussian for each sound+target
        print('choosing sounds {}'.format(n_sounds_list))
        file.write('\n Sounds = ')
        for item in n_sounds_list:
            file.write(' - ')
            file.write(str(item))
        # now random sample one distractor per each sound, so that they are drawn from the same distribution
        distr_list = []
        for i in n_sounds_list:
            # find gaussian
            class_idx = int(i/n_elem)
            elem_idx = i%n_class
            flag = 1
            while flag == 1:
                elem_list = list(np.arange(n_elem))
                elem_list.remove(elem_idx)
                new_distr = np.random.choice(elem_list)+class_idx*n_elem    
                if new_distr not in n_sounds_list:
                    flag = 0
                    print('found distr for sound {} -> {}'.format(i,new_distr))
            distr_list.append(new_distr)
            
        file.write('\n Distractors = ')
        for item in distr_list:
            file.write(' - ')
            file.write(str(item))
        
        # extract and concatenate the different mixtures
        source = np.zeros((n_in,0))
        mix_len = []
        if all_comb==False:
            for idx in range(n_mixt):
                print('combining: target = {} ; idx = {}'.format(n_sounds_list[0],n_sounds_list[idx+1]))
                mel = dataset[n_sounds_list[0]]+dataset[n_sounds_list[idx+1]]
                mel = mel.flatten()
                mel = np.expand_dims(mel, axis=1) 
                source = np.concatenate((source,mel),axis = 1)
                mix_len.append(np.shape(mel)[1])
        elif all_comb:
            for idx_i,sound_i in enumerate(n_sounds_list):
                for idx_j,sound_j in enumerate(n_sounds_list[idx_i+1:]):
                    print('combining: sound_i = {} ; sound_j = {}'.format(sound_i,sound_j))
                    mel = dataset[sound_i]+dataset[sound_j]
                    mel = mel.flatten()
                    mel = np.expand_dims(mel, axis=1)
                    source = np.concatenate((source,mel),axis = 1)
                    mix_len.append(np.shape(mel)[1])
        
        plt.figure()
        librosa.display.specshow(source, x_axis='time', y_axis='mel')
        plt.colorbar()
        plt.title('Mel frequency spectrogram for one sequence of mixtures')
        plt.savefig(save_path+exp_name+'_'+str(sim_n)+'_mel_mixt.pdf', fmt='pdf', dpi=350)
        #librosa.display.waveplot(source, sr=sample_rate)
        
        # save the training source for testing
        source_train = source
        
        # repeat the block n_rep times
        source = np.tile(source,n_rep_train)
        source_len = np.shape(source)[1]
        
        # min and max of source
        rate_min = np.amin(source)
        rate_max = np.amax(source)
        # normalize source
        source = (source-rate_min)/(rate_max-rate_min)
        # compute new min and max
        rate_min = np.amin(source)
        rate_max = np.amax(source)
        
        # training
        print("")
        print("***********")
        print("Learning... ")
        print("***********")
        
        sim_steps = np.shape(source)[1]*t_steps # all the "columns" of the spectrograms x how many time steps per column
        raster_t = []
        raster_i = []
        perc = np.linspace(0,sim_steps,11)
        perc = [int(x) for x in perc]
        
        start = timeit.default_timer()
        for i in range(sim_steps):
        #for i in range(2000):
            if i in perc:
                p = int(i/sim_steps*100)
                print("Step {}/{}. {}% advancement".format(i,sim_steps,p))
                #print("min w = {} , max w = {}".format(np.min(w),np.max(w)))        
            frame = i//t_steps # compute which time frame of the source spectrogram we use to compute firing rate        
            rate_in = source[:,frame]        
            rate_in = (rate_in-rate_min)/(rate_max-rate_min)*gain #normalized mixtures    
            prate = dt*rate_in*(10**-3) #prob of spikes of input neurons    
            id = np.random.rand(n_in)<prate #input neurons that spikes.        
            spiking_n = [i for i, x in enumerate(id) if x]
            if not not spiking_n:
                for t, s_n in enumerate(spiking_n):
                    raster_t.append(i)
                    raster_i.append(s_n)        
            I_syn = (1.0 - dt / tau_syn) * I_syn
            I_syn[id]+=1/tau/tau_syn
            PSP = (1.0 - dt / tau) * PSP + I_syn
            PSP_unit=PSP*24.5 # PSPs are normalized.
            V_dend = np.dot(w.T,PSP_unit) #voltage of dendrite    
            V_som_list = np.roll(V_som_list, -1,axis=1)
            V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som+np.dot(-w_inh,f)) #voltage of  soma
            V_som_list[:,-1] = V_som
            
            if record_w and i%w_change_gap == 0:
                for n_out in range(N):
                    w_lists[n_out] = np.roll(w_lists[n_out], -1, axis = 0)
                    w_lists[n_out][:,-1] = w[:,n_out]
        
            if i>t0:
                f = np.clip(g((V_som-np.mean(V_som_list,axis=1)) / np.std(V_som_list,axis=1))+np.random.randn(N)*noise,0,1) #noisy firing rates are used during training.
                w += eps  *np.outer((f-g(V_dend*g_d/(g_d+g_L))) , PSP_unit).T*beta*(1-g(V_dend*g_d/(g_d+g_L))) #the main learning rule
                w-=eps*w*0.5
                
                if plastic_inhib:
                    A_pre_p = (1.0 - dt / tau_p) * A_pre_p
                    A_pre_d = (1.0 - dt / tau_d) * A_pre_d
                    A_post_p = (1.0 - dt / tau_p) * A_post_p
                    A_post_d = (1.0 - dt / tau_d) * A_post_d
        
                    spike_id=np.random.rand(N)<f*gain*(10**-3)
                    A_pre_p[spike_id]+=cA_p
                    A_pre_d[spike_id]+=cA_d
                    A_post_p[spike_id]+=cA_p
                    A_post_d[spike_id]+=cA_d
        
                    w_inh[spike_id,:]+=np.matlib.repmat(((A_pre_p+A_pre_d)*w_inh_max),sum(spike_id),1)
                    w_inh[:,spike_id]+=np.matlib.repmat((A_post_p+A_post_d)*w_inh_max,sum(spike_id),1).T
                    
                    w_inh*=mask
                    w_inh[w_inh<0] = 0
                    w_inh[w_inh>w_inh_max] = w_inh_max
                    
                if record_w and i%w_change_gap == 0:
                    sigma_sum = 0
                    for n_out in range(N):
                        sigma_sum += np.sum(np.std(w_lists[n_out],axis=1))
                    w_change.append(sigma_sum/N/n_in)
                    
        stop = timeit.default_timer()
        print('Time for training: ', stop - start) 
            
        """      
        plt.figure()
        colors = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"]
        for rep in range(2):
            for track in range(n_mixt):
                start = (track*mix_len[0]+rep*n_mixt*mix_len[0])*t_steps
                end = ((track+1)*mix_len[0]+rep*n_mixt*mix_len[0])*t_steps-1
                idx = [x for x in np.arange(len(raster_t)) if (raster_t[x]>=start and raster_t[x]<end)]
                spike_i = [raster_i[x] for x in idx]
                spike_t = [raster_t[x] for x in idx]
                #print(start,end)
                plt.plot(spike_t,spike_i,'.',color = colors[track])
        plt.xlabel("Time")
        plt.ylabel("Neuron indices")
        #plt.xlim([25000,25100])
        plt.show()
        """
        
        if record_w:
            plt.figure()
            plt.plot(w_change,'.')
            plt.xlabel('time steps / {}'.format(w_change_gap))
            plt.ylabel('weight change')
            plt.title('Weight change with {} repetitions'.format(n_rep_train))
            plt.savefig(save_path+exp_name+'_'+str(sim_n)+'_w_change.pdf', fmt='pdf', dpi=350)
            
            filename = exp_name+'_w_change.txt'
            np.savetxt(save_path+filename, w_change)
        
        # build audio source for inference
        # extract and concatenate the sounds presented during learning and their distractors
        source = np.zeros((n_in,0)) 
        source_wave = []     
        mix_len = []
        idx_w = 1
        for idx in range(n_sounds_tot):
            print('adding sound = {}'.format(n_sounds_list[idx]))
            mel = dataset[n_sounds_list[idx]]
            mel = mel.flatten()
            mel = np.expand_dims(mel, axis=1) 
            source = np.concatenate((source,mel),axis = 1)
            mix_len.append(np.shape(mel)[1])
            wave = np.ones(t_steps)*(idx_w+1)
            source_wave = np.concatenate((source_wave,wave))
            idx_w += 1
        idx_w = 1.2
        for idx in range(len(distr_list)):
            print('adding distractor = {}'.format(distr_list[idx]))
            # extract the mel of the distractor and that of the sound
            mel_d = dataset[distr_list[idx]]
            mel_s = dataset[n_sounds_list[idx]]
            if plot_distr:
                fig = plt.figure()
                plt.subplot(1,3,1)
                plt.imshow(mel_d)
                plt.xlabel('Time frame')
                plt.ylabel('Frequency band')
                plt.title('original distractor {}'.format(distr_list[idx]))
            
            # set a time slice equal to one eight of the spectrogram to be equal to the target
            n_equal_frames = int(np.shape(mel_d)[1]/8)
            #print('how many', n_equal_frames)
            start_equal_frames = np.random.choice(np.arange(np.shape(mel_d)[1]-n_equal_frames))
            #print('from',start_equal_frames)
            for fr in range(start_equal_frames,start_equal_frames+n_equal_frames):
                mel_d[:,fr] = mel_s[:,fr]
            if plot_distr:
                plt.subplot(1,3,2)
                plt.imshow(mel_d)
                plt.xlabel('Time frame')
                plt.title('modified distractor')
                plt.subplot(1,3,3)
                plt.imshow(mel_s)
                plt.xlabel('Time frame')
                plt.title('associated sound {}'.format(n_sounds_list[idx]))
                plt.savefig(save_path+exp_name+'_'+str(sim_n)+'_distr_for_sound'+str(n_sounds_list[idx])+'.png')
            #print('modified frames {} to {}'.format(start_equal_frames,f))
            
            mel_d = mel_d.flatten()
            mel_d = np.expand_dims(mel_d, axis=1) 
            source = np.concatenate((source,mel_d),axis = 1)
            mix_len.append(np.shape(mel_d)[1])  
            wave = np.ones(t_steps)*(idx_w+1)
            source_wave = np.concatenate((source_wave,wave))
            idx_w += 1
        
        plt.figure()
        librosa.display.specshow(source, x_axis='time', y_axis='mel')
        plt.xlabel('time')
        plt.ylabel('mel')
        plt.title('Mel frequency spectrogram for one sequence for single sources')
        plt.savefig(save_path+exp_name+'_'+str(sim_n)+'_mel_single.pdf', fmt='pdf', dpi=350)
        
        # repeat the block n_rep times
        source = np.tile(source,n_rep_test)
        source_wave = np.tile(source_wave,n_rep_test)
        source_wave_len_test = np.shape(source_wave)[0]
        
        if test_mixt:
            source = np.concatenate((source,source_train),axis = 1)
        source_len_test = np.shape(source)[1]
        
        # min and max of source
        rate_min = np.amin(source)
        rate_max = np.amax(source)
        # normalize source
        source = (source-rate_min)/(rate_max-rate_min)
        # compute new min and max
        rate_min = np.amin(source)
        rate_max = np.amax(source)
        
        # testing
        print("")
        print("***********")
        print("Testing... ")
        print("***********")
        
        test_len = source_len_test*t_steps
        f_list = np.zeros((N,test_len*loop)) 
        
        out_lists = []
        for n_out in range(N):
            new_out_lists = np.zeros((loop,test_len))
            out_lists.append(new_out_lists)
            
        for j in range(loop):
            print("loop {} / {}".format(j,loop))
            PSP = np.zeros(n_in)
            I_syn = np.zeros(n_in)
            V_dend = np.zeros(N)
            V_som = np.zeros(N)
            id = np.zeros((test_len,n_in),dtype=bool)
            
            for i in range(test_len):
                frame = i//t_steps # compute which time frame of the source spectrogram we use to compute firing rate
                rate_in = source[:,frame]
                rate_in = (rate_in-rate_min)/(rate_max-rate_min)*gain #normalized mixtures
                prate = dt*rate_in*(10**-3) #prob of spikes of input neurons
                id[i,:] = np.random.rand(n_in)<prate
                I_syn = (1.0 - dt / tau_syn) * I_syn
                I_syn[id[i,:]]+=1/tau/tau_syn
                PSP = (1.0 - dt / tau) * PSP + I_syn
                PSP_unit=PSP*24.5
                V_dend = np.dot(w.T,PSP_unit)
                V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som+np.dot(-w_inh,f))
                f = g(V_som)
                
                for n_out in range(N):
                    out_lists[n_out][j,i]=f[n_out]
        
        # plot results
        # a. full test
        nr_sub = 1+N
        # plot source sound during testing
        plt.figure(figsize=(14,16))
        plt.subplot(nr_sub,1,1)
        colors = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","orange","green","red","purple","brown","pink","gray","olive","cyan","orange","green","red","purple","brown","pink","gray","olive","cyan"]
        
        time_test = np.arange(source_wave_len_test)
        for rep in range(n_rep_test):
            for im in range(n_sounds_tot*2):
                start = im*t_steps+rep*n_sounds_tot*2*t_steps
                end = (im+1)*t_steps+rep*n_sounds_tot*2*t_steps
                
                plt.plot(time_test[start:end],source_wave[start:end],color = colors[im])
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        if test_mixt:
            plt.plot(time_test[end:time_test[-1]],source_wave[end:time_test[-1]],color = 'k')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")    
        end_sound = np.arange(0,test_len,step=t_steps)
        colors = ["tomato","turquoise","dodgerblue","yellowgreen","mediumpurple","deeppink","palegreen","orange","lightgrey","royalblue","gold","sandybrown","limegreen"]
        for pl in range(nr_sub-1):
            #col = colors[pl]
            plt.subplot(nr_sub,1,2+pl)
            plt.plot(np.mean(out_lists[pl],axis=0),color = colors[pl])
            #plt.xaxis.set_ticks_position('bottom')
            #plt.yaxis.set_ticks_position('left')
            #plt.spines['right'].set_color('none')
            #plt.spines['top'].set_color('none')
            #plt.xaxis.set_major_locator(pl.NullLocator())    
            plt.ylabel("Activity n "+str(pl+1), fontsize=11)
            #plt.xlim([100,200])
            for x in end_sound:
                plt.axvline(x,c = 'k',linestyle='--',linewidth = 0.4)    
        plt.xlabel("Time [ms]", fontsize=11)
        #fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95,hspace=0.3)
        plt.savefig(save_path+exp_name+'_'+str(sim_n)+'_outputs.png')
        
        
        # b. two test sequences
        nr_sub = 1+N
        # plot source sound during testing
        plt.figure(figsize=(14,16))
        plt.subplot(nr_sub,1,1)
        colors = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","orange","green","red","purple","brown","pink","gray","olive","cyan","orange","green","red","purple","brown","pink","gray","olive","cyan"]
        
        time_test = np.arange(source_wave_len_test)
        for rep in range(2):
            for im in range(n_sounds_tot*2):
                start = im*t_steps+rep*n_sounds_tot*2*t_steps
                end = (im+1)*t_steps+rep*n_sounds_tot*2*t_steps
                
                plt.plot(time_test[start:end],source_wave[start:end],color = colors[im])
        plt.xlim((0,t_steps*(n_sounds+1)*2*2)) 
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        if test_mixt:
            plt.plot(time_test[end:time_test[-1]],source_wave[end:time_test[-1]],color = 'k')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")    
        end_sound = np.arange(0,t_steps*n_sounds_tot*2*2,step=t_steps)
        colors = ["tomato","turquoise","dodgerblue","yellowgreen","mediumpurple","deeppink","palegreen","orange","lightgrey","royalblue","gold","sandybrown","limegreen"]
        for pl in range(nr_sub-1):
            #col = colors[pl]
            plt.subplot(nr_sub,1,2+pl)
            plt.plot(np.mean(out_lists[pl],axis=0),color = colors[pl])
            #plt.xaxis.set_ticks_position('bottom')
            #plt.yaxis.set_ticks_position('left')
            #plt.spines['right'].set_color('none')
            #plt.spines['top'].set_color('none')
            #plt.xaxis.set_major_locator(pl.NullLocator())    
            plt.ylabel("Activity n "+str(pl+1), fontsize=11)
            #plt.xlim([100,200])
            for x in end_sound:
                plt.axvline(x,c = 'k',linestyle='--',linewidth = 0.4)
            plt.xlim((0,t_steps*(n_sounds+1)*2*2)) 
        plt.xlabel("Time [ms]", fontsize=11)
        #fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95,hspace=0.3)
        plt.suptitle('Sounds: {} Distractors: {}'.format(n_sounds_list,distr_list))
        plt.savefig(save_path+exp_name+'_'+str(sim_n)+'_outputs_2test.png')
        
        # save the outputs
        ### weights ###
        filename = exp_name+'_'+str(sim_n)+'_weights.txt'
        np.savetxt(save_path+filename, w)
        print("weights saved")
        ### v trace ###
        for pl in range(nr_sub-1):
            filename = exp_name+'_'+str(sim_n)+'activity_n'+str(pl+1)+'.txt'
            np.savetxt(save_path+filename, np.mean(out_lists[pl],axis=0))
        print("outputs saved")
        
        # COMPUTE PERFORMANCE OF THE MODEL
        n_sounds = len(n_sounds_list)
        n_distr = len(distr_list)
        output_list = []
        for n in range(N):
            out = np.loadtxt(save_path+exp_name+'_'+str(sim_n)+'activity_n'+str(n+1)+'.txt')
            output_list.append(out)
        if test_mixt:
            n_rep_tot=n_rep_test+1
        else:
            n_rep_tot=n_rep_test
            
        if test_mixt==False:
            test_len = len(output_list[0])
        else:
            test_len = len(output_list[0])-n_mixt*t_steps # length of the test on single sounds
            for n in range(N):
                output_list[n] = output_list[n][0:test_len]
        # build the set of points without the distractor
        X = []
        y = []
        
        sound_list = n_sounds_list+distr_list
        print(sound_list)
        #sound_list = ['siren','trumpet','cymbals']
        decay = 50
        true_lab = []
        for rep in range(n_rep_test):
        #for rep in range(1):
            for s in range(n_sounds):
                start = s*t_steps + rep*(n_sounds+n_distr)*t_steps
                end = start+t_steps
                for idx in range(start+decay,end):
                    new_point = []
                    for n in range(N):
                        m = output_list[n][idx]
                        new_point.append(m)
                    X.append(new_point)
                    y.append(sound_list[s])
                true_lab.append(s)
                #print(s,new_point)
        X = np.array(X)
        y = np.array(y)
        print(np.shape(X))
        
        # build the set of points for the distractors
        Xd = []
        decay = 50
        for rep in range(n_rep_test):
        #for rep in range(1):
            for s in range(n_sounds,n_sounds+n_distr):
                start = s*t_steps + rep*(n_sounds+n_distr)*t_steps
                end = start+t_steps
                for idx in range(start+decay,end):
                    new_point = []
                    for n in range(N):
                        m = output_list[n][idx]
                        new_point.append(m)
                    Xd.append(new_point)
                #print(s,new_point)
        Xd = np.array(Xd)
        np.shape(Xd)
        # apply PCA on targets and plot
        pca = PCA()
        pca.fit(X)
        X_PCA = pca.transform(X)
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        palette = sns.color_palette("bright", n_sounds)
        plt.figure(figsize=(8,6))
        g_fig = sns.scatterplot(X_PCA[:,0], X_PCA[:,1], hue=y, legend='full', palette=palette)
        fig = g_fig.get_figure()
        ax = plt.gca()
        ax.set_title('Experiment '+exp_name)
        fig.savefig(save_path+exp_name+'_'+str(sim_n)+'_PCA_for_GMM.png')
        # apply GMM on targets and plot
        gmm = GaussianMixture(n_components=n_sounds).fit(X_PCA[:,0:2])
        labels_GMM = gmm.predict(X_PCA[:,0:2])
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        palette = sns.color_palette("bright", len(Counter(labels_GMM).keys()))
        plt.figure(figsize=(8,6))
        g_fig = sns.scatterplot(X_PCA[:,0], X_PCA[:,1], hue=labels_GMM, legend='full', palette=palette)
        fig = g_fig.get_figure()
        ax = plt.gca()
        ax.set_title('PCA -'+exp_name)
        fig.savefig(save_path+exp_name+'_'+str(sim_n)+'_PCA.png')
        # save log likelihood of targets
        prob_targets = gmm.score_samples(X_PCA[:,0:2])
        
        # apply PCA and GMM on distractors (same transformations as targets) and plot
        Xd_PCA = pca.transform(Xd)
        labels_d = gmm.predict(Xd_PCA[:,0:2])
        
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        palette = sns.color_palette("bright", len(Counter(labels_GMM).keys()))
        paletted = sns.color_palette("dark", len(Counter(labels_d).keys()))
        plt.figure(figsize=(8,6))
        g_fig = sns.scatterplot(X_PCA[:,0], X_PCA[:,1], hue=labels_GMM, legend='full', palette=palette)
        f = sns.scatterplot(Xd_PCA[:,0], Xd_PCA[:,1], hue=labels_d, legend='full', palette=paletted)
        fig = g_fig.get_figure()
        ax = plt.gca()
        ax.set_title('GMM -'+exp_name)
        fig.savefig(save_path+exp_name+'_'+str(sim_n)+'_GMM.png')
        # save log likelihood of distractors
        prob_distr = gmm.score_samples(Xd_PCA[:,0:2])
        
        # display predicted scores by the model as a contour plot
        x = np.linspace(-1.4, 1.7)
        y = np.linspace(-1, 1)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = gmm.score_samples(XX)
        Z = np.array([-z for z in Z])
        Z = Z.reshape(X.shape)
        
        fig = plt.figure(figsize=(6,4))
        CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                         levels=np.logspace(0, 3, 10))
        CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plt.scatter(X_PCA[:, 0], X_PCA[:, 1], .8)
        plt.scatter(Xd_PCA[:, 0], Xd_PCA[:, 1], .8,c = 'g')
        plt.title(exp_name +' - Negative log-likelihood predicted by GMM')
        plt.axis('tight')
        fig.savefig(save_path+exp_name+'_'+str(sim_n)+'_GMM_log_L.png')
        
        # add the log likelihood for targets and distractors to the list
        prob_targets_all = np.concatenate((prob_targets_all,prob_targets))
        prob_distr_all = np.concatenate((prob_distr_all,prob_distr))
        
        # compute the auc of the roc curve for this run
        n_targets = np.shape(prob_targets)[0]
        lab_targets = np.ones(n_targets)
        n_distr = np.shape(prob_distr)[0]
        lab_distr = np.zeros(n_distr)
        true_labels = np.concatenate((lab_targets,lab_distr))
        pred = np.concatenate((prob_targets,prob_distr))
        LL_min = -100
        LL_max = 15
        pred = np.clip(pred, LL_min, LL_max)
        #bins = [-100,-15,-5,0,+15]
        bins = [-100,-20,-10,5,+15]
        bin_pred = np.zeros(np.shape(pred)[0])
        for idx,p in enumerate(pred):
            if p>bins[3] and p<bins[4]:
                bin_pred[idx] = 1.
            elif p>bins[2] and p<bins[3]:
                bin_pred[idx] = 0.66
            elif p>bins[1] and p<bins[2]:
                bin_pred[idx] = 0.33
            elif p>bins[0] and p<bins[1]:
                bin_pred[idx] = 0.
        fpr, tpr, thresholds = roc_curve(true_labels, bin_pred)
        auc = roc_auc_score(true_labels, bin_pred)
        auc = np.round(auc,3)
        print('At {} AUC: {}'.format(sim_n,auc))
        plt.figure()
        plt.plot(fpr, tpr, marker='.', label= str(sim_n))
        plt.savefig(save_path+exp_name+'_'+str(sim_n)+'ROC.png')
        auc_all.append(auc)
        
        file.write('\n AUC score = ')
        file.write(str(auc))
    
    auc_mean = np.mean(auc_all)
    auc_std = np.std(auc_all)
    print('AUC: mean={} , std={}'.format(auc_mean,auc_std))
    file.write('\n Final AUC score: ')
    file.write('mean = ')
    file.write(str(auc_mean))
    file.write('   -   std = ')
    file.write(str(auc_std))
    
    # save log likelihood
    filename = exp_name+'logL_targets.txt'
    np.savetxt(save_path+filename, prob_targets_all)
    filename = exp_name+'logL_distr.txt'
    np.savetxt(save_path+filename, prob_distr_all)
             
    file.close()
    

