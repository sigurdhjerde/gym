import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, stem, subplot, xlim, suptitle
# from gym.envs.cambridge_model.subject_stochastic import sample_patient
ion()
from gym.envs.registration import register


# def test_parameters(n, init_basal=6.43):
def test_parameters(pars, init_basal=6.43):
    np.random.seed(0)
    env = gym.make('HovorkaCambridge-v0')

    env.env.reset_basal_manually = init_basal

    # pars = np.load('parameters.npy')
    # pars = np.load('parameters_hovorka_fixed.npy')
    # pars = np.load('parameters_hovorka.npy')
    # pars = np.load('parameters_hovorka_new.npy')
    # pars = np.load('parameters_hovorka_new_min_8.npy')
    # pars = np.load('parameters_cambridge.npy')

    # Loading optimal basal
    # optimal_basal = np.load('optimal_basal.npy')
    # optimal_basal = np.load('optimal_basal_min_8.npy')
    # optimal_basal = optimal_basal[n]
    optimal_basal = init_basal

    # Plotting episodes for all sampled patients
    # figure()
    # env.env.P = pars[:, n]
    env.env.P = pars
    # env.env.reset_basal_manually = optimal_basal
    env.env.bolus = 25
    env.reset()

    for j in range(48):
        s, r, d, i = env.step(np.array([optimal_basal]))

    # figure()
    plot(env.env.bg_history)
    ylim(0, 600)

    # env.reset()

    # for j in range(48):
        # s, r, d, i = env.step(np.array([0]))

    # plot(env.env.bg_history)
    # legend(['optimal basal insulin', 'zero insulin'])
    title('Basal bolus control' + ' ' + str(optimal_basal))
    show()


if __name__ == '__main__':

    test_parameters(0)
