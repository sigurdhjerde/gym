import numpy as np
import gym

def hovorka_cambridge_pars(pat_num):
    '''
    Loading and returning cambridge parameters
    '''
    # pars =  np.load(gym.__path__[0] + '/envs/cambridge_model/parameters_hovorka.npy')
    # init_basal_rates = np.load(gym.__path__[0] + '/envs/cambridge_model/init_basal.npy')

    # pars =  np.load(gym.__path__[0] + '/envs/cambridge_model/stochastic_subjects/parameters_hovorka_new.npy')
    # init_basal_rates = np.load(gym.__path__[0] + '/envs/cambridge_model/stochastic_subjects/optimal_basal.npy')

    pars =  np.load(gym.__path__[0] + '/envs/diabetes/optimal_parameters/pars.npy')
    init_basal_rates = np.load(gym.__path__[0] + '/envs/diabetes/optimal_parameters/optimal_basal.npy')
    optimal_bolus = np.load(gym.__path__[0] + '/envs/diabetes/optimal_parameters/optimal_bolus.npy')

    return pars[:, pat_num], init_basal_rates[pat_num], optimal_bolus[pat_num]
