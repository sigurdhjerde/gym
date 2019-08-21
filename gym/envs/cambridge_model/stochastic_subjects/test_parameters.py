import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, stem, subplot, xlim, suptitle
ion()
from gym.envs.registration import register


def test_parameters_max_min(P, init_basal=6.43, carb_factor=25):
    env = gym.make('HovorkaCambridge-v0')

    env.env.reset_basal_manually = init_basal

    # pars = np.load('parameters.npy')
    # pars = np.load('parameters_hovorka_fixed.npy')
    # pars = np.load('parameters_hovorka.npy')
    # pars = np.load('parameters_cambridge.npy')

    figure()
    env.env.P = P
    env.env.bolus = carb_factor
    env.reset()

    for j in range(48):
        s, r, d, i = env.step(np.array([0]))

    plot(env.env.bg_history)

    env.reset()

    for j in range(48):
        s, r, d, i = env.step(np.array([50]))

    plot(env.env.bg_history)
    legend(['zero insulin', 'max insulin'])
    title('Testing a particular set of parameters')
    show()


def test_parameters(env, P, init_basal=6.43, carb_factor=25):

    env.env.reset_basal_manually = init_basal

    env.env.P = P
    env.env.bolus = carb_factor
    env.reset()

    for j in range(48):
        s, r, d, i = env.step(np.array([init_basal]))

    # figure()
    plot(env.env.bg_history)
    # ylim(0, 600)

    # title('Testing a particular set of parameters')
    show()


if __name__ == '__main__':

    # test_parameters(0)
    print('hello')