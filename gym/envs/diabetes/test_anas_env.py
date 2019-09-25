'''
Minimal example to test if the anas patient works
Do not edit!
'''
import numpy as np
import gym

from gym.envs.registration import register

# =================
# Anas patients
# =================
pn = 0

register(
    id='AnasPatient-1-v0',
    entry_point='gym.envs.diabetes.anas_patient:AnasPatient',
    kwargs={'patient_number': pn}
   )

env = gym.make('AnasPatient-1-v0')

env.reset_basal_manually = env.init_basal
env.reset()

# Looping through env
for i in range(72):
    s, r, d, i = env.step(np.array([env.init_basal]))

env.render()
