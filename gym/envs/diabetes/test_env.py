import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, subplot

# from gym.envs.diabetes.hovorka_cambride_pars import hovorka_cambridge_pars
from gym.envs.diabetes.hovorka_model import hovorka_parameters

env = gym.make('HovorkaCambridge-v0')


# P = hovorka_parameters(70)
# env.env.P = P
# init_basal_optimal = 6.43
# env.env.init_basal_optimal = init_basal_optimal
# env.env.reset_basal_manually = init_basal_optimal

reward = []
bg = []
cgm = []
iob = []

# env.env.bolus = 25
env.reset()

for i in range(48):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([env.env.init_basal_optimal]))

    bg.append(env.env.simulation_state[4])
    cgm.append(env.env.simulation_state[-1] * env.env.P[12])
    reward.append(r)
    iob.append(s[-1])



# figure()
# # plot(env.env.bg_history)
# plot(bg)
# plot(cgm)
# legend(['bg', 'cgm'])
# title('bg and cgm')
# ion()
# show()
# figure()
subplot(2, 1, 1)
plot(env.env.bg_history)
subplot(2, 1, 2)
plot(env.env.meals)
# ylim(0, 300)
show()
title('Anas meals -- spike meal and bolus')


## Plotting iob and such
# figure()
# subplot(3, 1, 1)
# plot(iob)
# title('iob')

# subplot(3, 1, 2)
# plot(env.env.meals)
# title('meals')

# subplot(3, 1, 3)
# plot(env.env.meal_indicator)
# title('bolus meals')
# show()
