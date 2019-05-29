import numpy as np
import gym

def hovorka_cambridge_pars(pat_num):
    '''
    Loading and returning cambridge parameters
    '''
    # pars =  np.load(gym.__path__[0] + '/envs/cambridge_model/parameters_hovorka.npy')
    # init_basal_rates = np.load(gym.__path__[0] + '/envs/cambridge_model/init_basal.npy')

    # pars = np.load(gym.__path__[0] + '/envs/cambridge_model/pars.npy')
    # init_basal_rates = np.load(gym.__path__[0] + '/envs/cambridge_model/optimal_basal.npy')

    pars = np.load(gym.__path__[0] + '/envs/cambridge_model/p_matlab.npy')
    init_basal_rates = np.load(gym.__path__[0] + '/envs/cambridge_model/basal_matlab.npy')

    return pars[:, pat_num], init_basal_rates[pat_num]
