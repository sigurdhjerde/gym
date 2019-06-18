''' Selecting parameters and estimating constant initial
insulin using a crude grid search.
'''
import numpy as np
import gym
from gym.envs.cambridge_model.stochastic_subjects.subject_stochastic import sample_patient

env = gym.make('HovorkaCambridge-v0')

def parameter_test(env, P, basal_rate=6.43):

    env.env.reset_basal_manually = basal_rate
    env.reset()
    env.env.P = P
    env.env.bolus = 15e10

    for i in range(48):

        # Step for the minimal/hovorka model
        s, r, d, i = env.step(np.array([basal_rate]))

    p_too_high = False

    if any(env.env.bg_history > 700):
        p_too_high = True

    not_converging = False

    if env.env.bg_history[1200] > 150:
        not_converging = True

    no_meal_spike = False

    if env.env.bg_history[240] < 200:
        no_meal_spike = True

    return p_too_high, not_converging, no_meal_spike, env.env.bg_history


def compare_to_hovorka():
    '''
    Comparing to the hovorka default patient
    '''

    env = gym.make('HovorkaCambridge-v0')

    from gym.envs.diabetes.hovorka_model import hovorka_parameters

    P = hovorka_parameters(70)

    env.env.reset_basal_manually = 6.43

    env.env.bolus = 1e10

    env.env.P = P

    env.reset()

    for i in range(48):
            s, r, d, i = env.step(np.array([6.43]))

    hovorka_curve = env.env.bg_history

    mse = np.zeros(500)

    for i in range(500):
        mse[i] = ((hovorka_curve - bg_hist[:, i])**2).mean()

    mse_sorted = np.argsort(mse)

    good_patients = mse_sorted[np.random.choice(200, 30)]

    return hovorka_curve, good_patients


def check_initial_insulin_bg(env, P):
    ''' Grid search to check the
    basal insulin needed to keep the
    virtual patient at 6 mmol/l
    '''

    insulin_vector = np.linspace(2, 100, 100)

    S = []

    for i in insulin_vector:
        env.env.reset_basal_manually = i
        env.env.P = P
        s = env.reset()
        S.append(s[0])

    idx = (np.abs(np.array(S) - 108)).argmin()

    return insulin_vector[idx]



def generate_parameters(num_pars):
    t = int(0)

    pars = np.zeros([18, num_pars])
    bg_hist = np.zeros([1440, num_pars])
    bw = np.zeros([num_pars])
    optimal_basal = np.zeros([num_pars])

    while t < num_pars:

        print('iteration ' + str(t))
        # Generating a random sample
        P, BW = sample_patient()

        # Cheking the parameters
        ob = check_initial_insulin_bg(env, P)
        p_too_high, not_converging, no_meal_spike, bg = parameter_test(env, P, ob)
        p_too_low = np.any(np.array(P)<0)

        # If insulin absorption is too slow, cut!
        # ins_ab_too_slow = P[1]>90

        # if not (p_too_high or p_too_low or ob<8 or ins_ab_too_slow):
        if not (p_too_high or p_too_low or ob<8 or not_converging):
        # if not (p_too_high or p_too_low or ob<8 or ins_ab_too_slow or not_converging):
        # if not (p_too_low or ob<8):
            pars[:, t] = P
            bw[t] = BW
            optimal_basal[t] = ob
            bg_hist[:, t] = bg
            t += 1

    return pars, bw, optimal_basal, bg_hist

def calculate_carb_factor(P):
    ''' Calculate the carb factor as Anas is doing it
    '''
    MCHO = 180
    St = P[5]/P[4]
    Sd = P[7]/P[6]
    basal_glucose = 6
    Vg = P[12]
    ke = P[10]
    Vi = P[11]

    carb_factor = (MCHO * (1.4 * max(St, 16e-4) + 0.6 * min(max(Sd, 3e-4), 12e-4)) * basal_glucose * Vg) / (ke * Vi)
    
    return carb_factor


# init_basal_range = np.linspace(6, 8, 20)

## Doing the grid search to determine initial value
if __name__ == '__main__':

    # print('main')
    pars, bw, optimal_basal, bg_hist = generate_parameters(50)
    # np.save('parameters_hovorka_new_min_8', pars)
    # np.save('parameters_hovorka_new_min_8_bw', bw)
    # np.save('optimal_basal_min_8', optimal_basal)

