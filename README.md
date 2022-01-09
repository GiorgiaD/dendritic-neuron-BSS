# dendritic-neuron-BSS
Code to run the experiment 1 of the paper:
### Learning in Deep Neural Networks Using a Biologically Inspired Optimizer

Giorgia Dellaferrera, Toshitake Asabuki, Tomoki Fukai


# Requirements
We run the experiments with the following:

Numpy framework: Python 3.6.5, Numpy 1.17.3

Libraries: 

librosa 0.7.1, scipy 1.1.0, sklearn 0.19.1, matplotlib 2.2.2, pylab, pydub, 

# Experiments  
The main experiments are run through `Recovering_sound_sources_matlab_mult.py`. 

For example, to run with the standard settings:
```
python Recovering_sound_sources_matlab_mult.py --exp_name Experiment1 \
    --n_sounds 2 --N 8 
``` 

Substitute `--n_sounds 2` with `--n_sounds 3` to run the experiment with a larger number of mixtures. 

Substitute `--N 8` with `N 12` to run the experiment with a larger number of output neurons. 

Add `--all_comb` to run the experiment in the "all combination" set up. 

Add `--sparse_connectivity` to modify the network from fully connected to sparse connectivity. 


