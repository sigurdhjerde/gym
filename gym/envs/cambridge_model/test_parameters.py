import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, stem, subplot, xlim, suptitle
from gym.envs.cambridge_model.subject_stochastic import sample_patient
ion()
from gym.envs.registration import register


def test_parameters(n, init_basal=6.43):
    env = gym.make('CambridgeAbsolute-v0')

    env.env.reset_basal_manually = init_basal

    # pars = np.load('parameters.npy')
    # pars = np.load('parameters_hovorka_fixed.npy')
    pars = np.load('parameters_hovorka.npy')
    # pars = np.load('parameters_cambridge.npy')


    # Plotting episodes for all sampled patients
    figure()
    env.env.P = pars[:, n]
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


if __name__ == '__main__':

    # test_parameters(0)
    print('hello')
