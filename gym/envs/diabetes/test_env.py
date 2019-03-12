import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, subplot

# from gym.envs.diabetes.hovorka_cambride_pars import hovorka_cambridge_pars
from gym.envs.diabetes.hovorka_model import hovorka_parameters

# env = gym.make('HovorkaGaussian-v0')
# env = gym.make('HovorkaGaussian-v0')
# np.random.seed(0)
env = gym.make('HovorkaCambridge-v0')

P = hovorka_parameters(70)
env.env.P = P
init_basal_optimal = 6.43
env.env.init_basal_optimal = init_basal_optimal
env.env.reset_basal_manually = init_basal_optimal

# env = gym.make('HovorkaGaussianInsulin-v0')
# env = gym.make('HovorkaBinary-v0')
# env = gym.make('HovorkaAbsolute-v0')

# ==============
# Spike meals and bolus
# ==============
# Manual meal manipulation! Comparison to Anas El Fathi stuff
# env.env.eating_time = 1
# meal_amount = np.array([40, 80, 60, 30])  #+ np.random.choice(np.linspace(-15, 15, 7, dtype=int), 4)
# meal_times = np.array([8*60, 12*60, 18*60, 22*60]) + np.random.choice(np.linspace(-30,30,3, dtype=int), 4)

# meal_amount = np.array([40, 80, 60, 30])
# env.env.meals = np.zeros(1440)
# env.env.meals[[8*60, 12*60, 18*60, 22*60]] = np.array([40, 80, 60, 30])
# env.env.meals = env.env.meals * 1000/180
#
# env.env.meal_indicator = np.zeros(1440)
# env.env.meal_indicator[[8*60, 12*60, 18*60, 22*60]] = np.array([40, 80, 60, 30]) + np.random.randint(-20, 20)
# env.env.meal_indicator = env.env.meal_indicator * 1000/180


# env.env.meals = np.zeros(1440)
# env.env.meals[meal_times] = meal_amount
# env.env.meals = env.env.meals * 1000/180

# env.env.meal_indicator = np.zeros(1440)
# env.env.meal_indicator[meal_times] = meal_amount + np.random.randint(-20, 20)
# env.env.meal_indicator = env.env.meal_indicator * 1000/180

# =================================
# 30 minute meals and spike bolus TODO: no noise atm
# =================================
# env.env.eating_time = 30
# env.env.meals = np.zeros(1440)
# env.env.meal_indicator = np.zeros(1440)

# for i in range(4):
    # env.env.meals[meal_times[i]:meal_times[i]+29] = meal_amount[i] / 30
    # env.env.meal_indicator[meal_times[i]] = meal_amount[i]

# env.env.meals = env.env.meals * 1000/180
# env.env.meal_indicator = env.env.meal_indicator * 1000/180

# =================================
# 30 minute meals and 30 min bolus TODO: no noise atm
# =================================
# env.env.eating_time = 30
# env.env.meals = np.zeros(1440)
# env.env.meal_indicator = np.zeros(1440)

# for i in range(4):
    # env.env.meals[meal_times[i]:meal_times[i]+30] = meal_amount[i] / 30
    # env.env.meal_indicator[meal_times[i]:meal_times[i]+30] = meal_amount[i] / 30

# env.env.meals = env.env.meals * 1000/180
# env.env.meal_indicator = env.env.meal_indicator * 1000/180

# basal = 0
# env.env.reset_basal_manually = 6.43

reward = []
bg = []
cgm = []
iob = []

# env.env.reset_basal_manually = 6.43
env.env.bolus = 25
# env.env.meals = np.zeros(1440)
# env.env.meal_indicator = np.zeros(1440)
env.reset()

for i in range(48):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([init_basal_optimal]))
    # s, r, d, i = env.step(np.array([0]))

    bg.append(env.env.simulation_state[4])
    cgm.append(env.env.simulation_state[-1] * env.env.P[12])
    reward.append(r)
    iob.append(s[-1])
    # print(r)

    # Step for the discrete Hovorka incremental
    # env.step(2)



# env.render()
# figure()
# # plot(env.env.bg_history)
# plot(bg)
# plot(cgm)
# legend(['bg', 'cgm'])
# title('bg and cgm')
# ion()
# show()
# figure()
plot(env.env.bg_history)
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
