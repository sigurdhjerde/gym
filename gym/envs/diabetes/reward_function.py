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
        alpha = 1
        beta = 1

        reward = - alpha*(abs(blood_glucose_level - bg_ref)) - beta * (abs(action[1]-action[0]))

    elif reward_flag == 'gaussian':
        ''' Gaussian reward function '''
        h = 30

        reward =  np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 /h**2)


    return reward
