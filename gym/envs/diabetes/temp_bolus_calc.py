'''
Iterating through all environments to check what bolus rate is sufficient
'''
import numpy as np
from gym.envs.diabetes.hovorka_cambride_pars import hovorka_cambridge_pars
import gym

env = gym.make('HovorkaGaussian-v0')

# env.env.meals = np.zeros(1440)
# env.env.meal_indicator = np.zeros(1440)
env.env.reset_basal_manually = 6.43


def run_episode(basal):
    env.reset()
    for i in range(48):
        env.step(np.array([basal]))


