import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, subplot

from gym.envs.diabetes.hovorka_model import hovorka_parameters

env = gym.make('HovorkaCambridge-v0')

init_basal_optimal = 6.43

reward = []
bg = []
cgm = []
iob = []

env.reset()

for i in range(72):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([init_basal_optimal]))

    # Saved for possible printing
    bg.append(env.env.simulation_state[4])
    cgm.append(env.env.simulation_state[-1] * env.env.P[12])
    reward.append(r)
    iob.append(s[-1])


env.render()
