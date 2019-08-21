''' Selecting parameters and estimating constant initial
insulin using a crude grid search.
'''
import numpy as np
import gym

from gym.envs.cambridge_model.stochastic_subjects.subject_stochastic import sample_patient 
from pylab import plot, figure, title, show, ion, legend, ylim, stem, subplot, xlim, suptitle

# Loading environment
env = gym.make('HovorkaCambridge-v0')

# Changing to a single meal scenario
# env.env.meals = np.zeros(1440)
# env.env.meal_indicator = np.zeros(1440)

# meal_time = 120
# meal_amount = 90

# env.env.meals[120] = meal_amount * 1000 / 180
# env.env.meal_indicator[120] = meal_amount * 1000 / 180

def check_initial_insulin_bg(env, P):
    ''' Grid search to check the
    basal insulin needed to keep the
    virtual patient at 6 mmol/l
    '''

    insulin_vector = np.linspace(5, 100, 100)

    S = []

    for i in insulin_vector:
        env.env.reset_basal_manually = i
        env.env.P = P
        s = env.reset()
        S.append(s[0])

    idx = (np.abs(np.array(S) - 108)).argmin()

    return insulin_vector[idx]


def calculate_within_range(bg):
    '''
    Calculate within range

    The results is given in percentage
    '''

    total_time = len(bg)

    time_below = np.sum(bg<70)
    time_above = np.sum(bg>180)

    time_within_range = 1 - (time_below + time_above) / total_time

    time_hypo = 100 * (time_below / total_time)
    time_hyper = 100 * (time_above / total_time)

    return time_within_range, time_hypo, time_hyper

  

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


def generate_parameters(num_pars):
    t = int(0)

    pars = np.zeros([18, num_pars])
    # bg_hist = np.zeros([1440, num_pars])
    bw = np.zeros([num_pars])
    optimal_basal = np.zeros([num_pars])
    # carb_factors = np.zeros([num_pars])

    while t < num_pars:

        print('iteration ' + str(t))
        # Generating a random sample
        P, BW = sample_patient()

        # Optimal basal rate
        ob = check_initial_insulin_bg(env, P)

        # Carb factor based on Anas' formula
        # cf = calculate_carb_factor(P)

        p_too_high, not_converging, no_meal_spike, bg = parameter_test(env, P, ob)

        # Are any of the parameters below zero?
        p_too_low = np.any(np.array(P)<0)

        # If insulin absorption is too slow, cut!
        # ins_ab_too_slow = P[1]>90

        # if not (p_too_high or p_too_low or ob<8 or ins_ab_too_slow):
        # if not (p_too_high or p_too_low or ob<8 or not_converging):
        # if not (p_too_high or p_too_low or ob<8 or ins_ab_too_slow or not_converging):

        if not ((p_too_low or not_converging or p_too_high) and ob>8):
            pars[:, t] = P
            bw[t] = BW
            optimal_basal[t] = ob
            # carb_factors[t] = cf
            # bg_hist[:, t] = bg
            t += 1

    
    return pars, bw, optimal_basal, #carb_factors, bg_hist


def compare_to_hovorka():
    '''
    Comparing to the hovorka default patient
    '''

    env = gym.make('HovorkaCambridge-v0')

    from gym.envs.diabetes.hovorka_model import hovorka_parameters

    P = hovorka_parameters(70)

    env.env.reset_basal_manually = 6.43

    env.env.bolus = 30

    env.env.P = P

    env.reset()

    for i in range(48):
            s, r, d, i = env.step(np.array([6.43]))

    hovorka_curve = env.env.bg_history

    # mse = np.zeros(500)

    # for i in range(500):
    #     mse[i] = ((hovorka_curve - bg_hist[:, i])**2).mean()

    # mse_sorted = np.argsort(mse)

    # good_patients = mse_sorted[np.random.choice(200, 30)]

    return hovorka_curve#, good_patients


def parameter_test(env, P, basal_rate=6.43):

    env.env.reset_basal_manually = basal_rate
    env.reset()
    env.env.P = P
    env.env.bolus = 15e10

    for i in range(48):

        # Step for the minimal/hovorka model
        s, r, d, _ = env.step(np.array([basal_rate]))

    p_too_high = False

    if any(env.env.bg_history > 700):
        p_too_high = True

    not_converging = False

    if env.env.bg_history[400] > 150:
        not_converging = True

    no_meal_spike = False

    if env.env.bg_history[240] < 200:
        no_meal_spike = True

    return p_too_high, not_converging, no_meal_spike, env.env.bg_history


def test_parameters(env, P, init_basal=6.43, carb_factor=20):
    '''
    Testing the chosen parameter set
    '''

    env.env.reset_basal_manually = init_basal

    env.env.P = P
    env.env.bolus = carb_factor
    env.reset()

    for j in range(48):
        s, r, d, i = env.step(np.array([init_basal]))

    # figure()
    plot(env.env.bg_history)
    # ylim(0, 600)

    # Time in range
    tir, _, _ = calculate_within_range(env.env.bg_history)

    title('Testing a particular set of parameters')
    show()

    return tir, env.env.bg_history


def calculate_optimal_bolus(env, P, basal_rate):
    ''' Calculating the optimal bolus, based on either
    time-in-range or size and convergence speed of 
    postprandial spike.
    '''

    bolus_rates = np.linspace(10, 30, 30)
    tir = np.zeros_like(bolus_rates)

    ind = 0
    for i in bolus_rates:
        env.env.bolus = i
        env.env.P = P
        env.reset()

        for i in range(48):
            env.step(np.array([basal_rate]))

        # ion()
        # figure()
        # plot(env.env.bg_history)
        # show()

        # Time-in-range
        tir[ind] = calculate_within_range(env.env.bg_history)[0]

        ind += 1

        # env.render()


    # Choosing the minimal value
    optimal_bolus_time_in_range = bolus_rates[np.argmax(tir)] 

    return optimal_bolus_time_in_range


if __name__ == '__main__':

    num_pars = 10

    # Generate some parameters
    pars, bw, optimal_basal = generate_parameters(num_pars)

    # Hard coding Anas' constrains on the last parameters
    pars[2, :] = .9 # Bioavailability
    pars[-3, :] = 1 / 12 # CGM delay
    pars[-1, :] = 11 # Renal clearance threshold


    # Test and plot parameters
    tir = []
    bg_hist = []
    for i in range(num_pars): 
        t_i_r, b_h = test_parameters(env, pars[:, i], optimal_basal[i], 20)
        tir.append(t_i_r)
        bg_hist.append(b_h)



    ylim(0, 600)
    title('All virtual patients')        

    # Plotting the best sets according to time-in-range
    # Also removed parameters that does not give a stable basal glucose (e.g. rising with constant insulin)

    figure()
    count = 0
    index = []
    for i in range(num_pars): 
        if tir[i]>.75 and (bg_hist[i][400] - bg_hist[i][0] < 5):
            test_parameters(env, pars[:, i], optimal_basal[i], 20) 
            count += 1
            index.append(i)

    ylim(0, 600)
    title('Only virtual pars with TIR over 75%')


    # Calculating optimal bolus
    # optimal_bolus = np.zeros(len(index))

    # for i in range(len(index)):
    #     optimal_bolus[i] = calculate_optimal_bolus(env, pars[:, index[i]], optimal_basal[index[i]])


    # Plotting the optimal bolus
    # figure()
    # bg_all = []
    # for i in range(len(index)):
    #     env.env.P = pars[:, index[i]]
    #     env.env.bolus = optimal_bolus[i]
    #     env.env.reset_basal_manually = optimal_basal[index[i]]
    #     env.reset()
    #     for j in range(48):
    #         env.step(np.array([optimal_basal[index[i]]]))
    #     bg_all.append(env.env.bg_history)
    #     plot(env.env.bg_history)


    # Save parameters
    # np.save('parameters_hovorka', pars)
    # np.save('parameters_hovorka_bw', bw)



