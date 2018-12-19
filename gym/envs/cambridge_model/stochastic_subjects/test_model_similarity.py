import numpy as np

# Plotting for the rendering
# import matplotlib.pyplot as plt

# Cambridge simulator
from gym.envs.cambridge_model.cambridge_model import cambridge_model, cambridge_model_tuple
from gym.envs.diabetes.hovorka_model import hovorka_model_tuple
from gym.envs.cambridge_model.subject import subject

# ODE solver stuff
from scipy.optimize import fsolve

for i in range(6):
    P = subject(i+1)
    X0 = fsolve(cambridge_model_tuple, np.zeros(11), args=(0,0, P))
    print('cambridge subject ' + str(i+1))
    print(str(X0[4]/P[12] * 18))

for i in range(6):
    P = subject(i+1)
    X0 = fsolve(hovorka_model_tuple, np.zeros(11), args=(0,0, P))
    print('Hovorka subject ' + str(i+1))
    print(str(X0[4]/P[12] * 18))
