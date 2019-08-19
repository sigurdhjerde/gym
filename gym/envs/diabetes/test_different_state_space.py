'''
Testing the different state space parameters

step time and solver time
'''
import numpy as np
import gym
import seaborn as sns
sns.set()

env = gym.make('HovorkaCambridge-v0')

init_basal_optimal = 6.35
env.env.init_basal_optimal = init_basal_optimal
env.env.reset_basal_manually = init_basal_optimal

env.env.step_time = 10
env.env.solver_time = 1

env.reset()

num_steps = int(1440/env.env.step_time)
for i in range(num_steps):
    s, r, d, i = env.step(np.array([init_basal_optimal]))


env.render()