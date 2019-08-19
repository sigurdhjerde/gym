'''
Minimal example to test if the anas patient works
Do not edit!
'''
import numpy as np
import gym
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/jonas/Documents/git/pg_diabetes/garage/')
from anas_patients.load_mcgill_patients import matlab_to_python

from gym.envs.registration import register
import time

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

# start = time.time()

# Looping through env
for i in range(48):
    s, r, d, i = env.step(np.array([env.init_basal]))

# end = time.time()
# print(end - start)

# Maual plotting and printing values -- for debugging purposes
print(env.init_basal)
print(env.bg_history[0])
plt.subplot(1, 2, 1)
plt.plot(env.bg_history)
# plt.show()

# Rendering
# start = time.time()
# env.render()
# end = time.time()
# print(end - start)

# ==============================================================
# Comparing to just changing the parameters of the Hovorka Model
# ==============================================================
env = gym.make('HovorkaCambridge-v0')

P, basal_rate, carb_factor, _ = matlab_to_python(pn)

env.env.init_basal = basal_rate
env.env.P = P
env.env.bolus = carb_factor

env.env.reset_basal_manually = basal_rate

env.reset()

# Looping through env
for i in range(48):
    s, r, d, i = env.step(np.array(env.env.init_basal))

print(env.env.init_basal)
print(env.env.bg_history[0])
plt.subplot(1, 2, 2)
plt.plot(env.env.bg_history)
plt.show()

# ==============================================================