''' Selecting parameters and estimating constant initial
insulin using a crude grid search.
'''
import numpy as np
import gym
from gym.envs.cambridge_model.subject_stochastic import sample_patient

env = gym.make('CambridgeAbsolute-v0')

def test_parameters(env, P):

    env.reset()
    env.env.P = P

    for i in range(48):

        # Step for the minimal/hovorka model
        s, r, d, i = env.step(np.array([6.43]))

    p_too_high = False

    if any(env.env.bg_history > 700):
        p_too_high = True


    return p_too_high


def check_initial_insulin_bg(env, P):
    ''' Grid search to check the
    basal insulin needed to keep the
    virtual patient at 6 mmol/l
    '''

    insulin_vector = np.linspace(2, 15, 100)

    S = []

    for i in insulin_vector:
        env.env.reset_basal_manually = i
        env.env.P = P
        s = env.reset()
        S.append(s[0])

    idx = (np.abs(np.array(S) - 108)).argmin()

    return insulin_vector[idx]





def generate_parameters():
    t = int(0)

    pars = np.zeros([18, 30])
    bw = np.zeros([30])

    while t < 30:

        print('iteration ' + str(t))
        # Generating a random sample
        P, BW = sample_patient()

        # p_too_high = test_parameters(env, P)
        p_too_high = np.any(np.array(P)<0)

        if not p_too_high:
            pars[:, t] = P
            bw[t] = BW
            t += 1

    return pars, bw


# init_basal_range = np.linspace(6, 8, 20)

## Doing the grid search to determine initial value
if __name__ == '__main__':

    print('main')
    # pars, bw = generate_parameters()
    # np.save('parameters_hovorka', pars)
   # np.save('parameters_hovorka_bw', bw)

