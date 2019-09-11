import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, subplot

from gym.envs.diabetes.hovorka_model import hovorka_parameters

env = gym.make('HovorkaCambridge-v0')

# P = hovorka_parameters(70)
# env.env.P = P
# env.env.bolus = 30
init_basal_optimal = 6.43
# env.env.init_basal_optimal = init_basal_optimal
# env.env.reset_basal_manually = init_basal_optimal


# env.env.meal_times = np.array([8*60, 12*60, 18*60, 22*60]) # + np.random.choice(np.linspace(-30,30,3, dtype=int), 4)


reward = []
bg = []
cgm = []
iob = []

env.reset()

for i in range(72):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([init_basal_optimal]))
    # s, r, d, i = env.step(np.array([0]))

    bg.append(env.env.simulation_state[4])
    cgm.append(env.env.simulation_state[-1] * env.env.P[12])
    reward.append(r)
    iob.append(s[-1])
    # print(r)


env.render()
