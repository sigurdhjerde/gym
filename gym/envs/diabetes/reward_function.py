import numpy as np

def calculate_reward(blood_glucose_level, reward_flag='absolute', bg_ref=90, action=None):
    """
    Calculating rewards for the given blood glucose level
    """

    if reward_flag == 'binary':
        ''' Binary reward function'''
        low_bg = 70
        high_bg = 120

        if np.max(blood_glucose_level) < high_bg and np.min(blood_glucose_level) > low_bg:
            reward = 1
        else:
            reward = 0

    elif reward_flag == 'binary_tight':
        ''' Tighter version of the binary reward function,
        the bounds are [-5, 5] around the optimal rate.
        '''
        low_bg = 85
        high_bg = 95

        if np.max(blood_glucose_level) < high_bg and np.min(blood_glucose_level) > low_bg:
            reward = 1
        else:
            reward = 0


    elif reward_flag == 'squared':
        ''' Squared cost function '''

        reward = - (blood_glucose_level - bg_ref)**2

    elif reward_flag == 'absolute':
        ''' Absolute cost function '''

        reward = - abs(blood_glucose_level - bg_ref)

    elif reward_flag == 'absolute_with_insulin':
        ''' Absolute cost with insulin constraint '''

        if action == None:
            action = [0, 0]

        # Parameters
        alpha = .7
        beta = 1 - alpha

        reward = - alpha*(abs(blood_glucose_level - bg_ref)) - beta * (abs(action[1]-action[0]))

    elif reward_flag == 'gaussian':
        ''' Gaussian reward function '''
        h = 30
        # h = 15

        reward =  np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 /h**2)

    elif reward_flag == 'gaussian_with_insulin':
        ''' Gaussian reward function '''
        h = 30
        # h = 15
        alpha = .7

        bg_reward =  np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 /h**2)
        insulin_reward =  -1/15 * action + 1

        reward = alpha * bg_reward + (1 - alpha) * insulin_reward


    return reward
