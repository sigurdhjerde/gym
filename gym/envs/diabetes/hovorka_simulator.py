# Importing relevant stuff
from scipy.integrate import ode
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

# Hovorka model files
# from hovorka_model_tuple import hovorka_model_tuple
# from hovorka_model import hovorka_model
# from hovorka_parameters import hovorka_parameters

# Time variables
t0 = 0
dt = 1
t_end = 1440

# insulin variables
basal_init = 6.66
# basal_init = 5.99


# pre meal bolus announcement
premeal_bolus_time = 30

# Patient and patient parameters
body_weight = 70
eating_time = 30

def meal_setup(n_days):
    ''' Setting meal parameters '''

    # TODO: add noise perturbation
    # Meal vector
    meals = np.zeros(t_end)
    meal_indicator = np.zeros(t_end)

    # Meal parameters -- Mosching/Bastani setup
    # meal_times = [8*60, 14*60, 19*60]
    # meal_times = [2*60, 8*60, 13*60]
    # meal_amounts = [40, 70, 70]

    # A single meal
    # meal_times = [8*60]
    # meal_amounts = [50]

    # meals[meal_times[0]:meal_times[0]+eating_time] = meal_amounts[0] / eating_time * 1000 / 180

    # No meal
    meal_times = [0]
    meal_amounts = [0]


    for i in range(len(meal_times)):
        meals[meal_times[i] : meal_times[i] + eating_time] = meal_amounts[i]/eating_time * 1000 /180
        meal_indicator[meal_times[i]-premeal_bolus_time:meal_times[i]] = meal_amounts[i] * 1000 / 180

    # Repeating if multiple days
    if n_days > 1:
        meals = np.ravel(np.matlib.repmat(meals, 1, n_days))

    return meals, meal_indicator

def carb_rate(IC):
    ''' Converting carb rate to mmol/min'''
    carb_ratio = IC / eating_time *1000 / 180
    return carb_ratio

def simulation_setup(n_days, basal_init):
    """
    Initializing the Hovorka simulation.
    """

    # Patient parameters
    P = hovorka_parameters(body_weight)

    # Initial parameters -- numeric values taken from Moschings thesis
    initial_pars = (basal_init, 0, P)

    # Initial value
    X0 = fsolve(hovorka_model_tuple, np.zeros(10), args=initial_pars)

    # Simulation setup
    integrator = ode(hovorka_model)
    integrator.set_integrator('vode', method='bdf', order=5)
    integrator.set_initial_value(X0, 0)

    # Time settings, starts at 0 and updates every minute
    times = range(t0, t_end * n_days, dt)

    # Meal setup
    meals = meal_setup(n_days)

    return X0, meals[0], integrator, times, P

def simulate_first_day(BR, IC):
    ''' Running the Hovorka simulation for the first day'''

    X0, meals, integrator, times, P = simulation_setup(1)

    # Insulin setup -- basal rate
    basal_rate = BR * 1000 / t_end

    # Total insulin rate (adding bolus rate)
    insulin_rate = basal_rate + (meals * IC)

    # Running the simulation
    blood_glucose_level = [X0[4]]

    for i in times[1:]:

        # Updating parameters
        integrator.set_f_params(insulin_rate[i], meals[i], P)

        # Integrating the ODE
        integrator.integrate(integrator.t+dt)

        # Appending the data TODO: preallocation?
        blood_glucose_level.append(integrator.y[4])

    # Converting to mg/dl
    blood_glucose_level = np.asarray(blood_glucose_level) * 18 / P[12]

    return blood_glucose_level, integrator.y

def simulate_episode(BR, IC):
    '''
    Running the Hovorka simulation for a single episode.

    The episode stops if BG is below zero or above 300, or
    the number of iterations (minutes) reaches 2880.
    '''

    max_iter = 2880
    bg_low_thresh = 0
    bg_high_thresh = 500

    X0, _, integrator, _, P = simulation_setup(1)

    # Insulin setup -- basal rate
    basal_rate = BR * 1000 / 1440

    # Total insulin rate (adding bolus rate)

    # TODO: meals and bolus!
    # insulin_rate = basal_rate + (meals * IC)
    insulin_rate = basal_rate

    # Running the simulation
    blood_glucose_level = [X0[4]]

    done = 0
    while not done:

        # Updating parameters
        integrator.set_f_params(insulin_rate, 0, P)

        # Integrating the ODE
        integrator.integrate(integrator.t+dt)

        # Appending the data TODO: preallocation?
        blood_glucose_level.append(integrator.y[4])

        if len(blood_glucose_level) > max_iter:
            done = 1

        if (blood_glucose_level[len(blood_glucose_level)-1] < bg_low_thresh
        or blood_glucose_level[len(blood_glucose_level)-1] > bg_high_thresh):
            done = 1


    # Converting to mg/dl
    blood_glucose_level = np.asarray(blood_glucose_level) * 18 / P[12]

    return blood_glucose_level, integrator.y

def simulate_one_step(basal_rate, simulator_state):
    '''
    Running the Hovorka simulation for a single step

    The episode stops if BG is below zero or above 300, or
    the number of iterations (minutes) reaches 2880.
    '''

    _, _, integrator, _, P = simulation_setup(1, 6.66)

    # Only basal rate at the moment from unit pr day to mU/min
    insulin_rate = basal_rate * 1000 / 1440

    # Running the simulation

    # initial value
    integrator.set_initial_value(simulator_state, 0)

    blood_glucose_level = [simulator_state[4]]

    # Updating parameters
    integrator.set_f_params(insulin_rate, 0, P)

    # Integrating the ODE
    integrator.integrate(integrator.t+dt)

    # blood glucose level
    blood_glucose_level = integrator.y[4]

    # Converting to mg/dl
    blood_glucose_level = np.asarray(blood_glucose_level) * 18 / P[12]

    return blood_glucose_level, integrator.y

def simulate_one_step_with_meals(BR, IC, num_iter, simulator_state):
    '''
    Running the Hovorka simulation for a single step

    The episode stops if BG is below zero or above 300, or
    the number of iterations (minutes) reaches 2880.
    '''

    _, _, integrator, _, P = simulation_setup(1, 6.66)

    meals, meal_indicator = meal_setup(1)

    # Only basal rate at the moment from unit pr day to mU/min
    basal_rate = BR * 1000 / 1440

    insulin_rate = basal_rate + (meal_indicator[num_iter] * IC)/premeal_bolus_time

    # Running the simulation

    # initial value
    integrator.set_initial_value(simulator_state, 0)

    # Updating parameters
    integrator.set_f_params(insulin_rate, meals[num_iter], P)

    # Integrating the ODE
    integrator.integrate(integrator.t+dt)

    # blood glucose level
    blood_glucose_level = integrator.y[4]

    # Converting to mg/dl
    blood_glucose_level = np.asarray(blood_glucose_level) * 18 / P[12]

    return blood_glucose_level, integrator.y

def simulate_one_day(BR, IC, simulator_state):
    ''' Running the Hovorka simulation for a single day'''

    _, meals, integrator, times, P = simulation_setup(1)

    # Insulin setup -- basal rate
    basal_rate = BR * 1000 / t_end

    # Total insulin rate (adding bolus rate)
    insulin_rate = basal_rate + (meals * IC)

    # Running the simulation
    integrator.set_initial_value(simulator_state, 0)

    blood_glucose_level = [simulator_state[4]]

    for i in times[1:]:

        # Updating parameters
        integrator.set_f_params(insulin_rate[i], meals[i], P)

        # Integrating the ODE
        integrator.integrate(integrator.t+dt)

        # Appending the data TODO: preallocation?
        blood_glucose_level.append(integrator.y[4])

    # Converting to mg/dl
    blood_glucose_level = np.asarray(blood_glucose_level) * 18 / P[12]

    return blood_glucose_level, integrator.y

def simulate(BR, IC, n_days):
    ''' Running the Hovorka simulation '''

    # Simulation setup
    X0, meals, integrator, times, P = simulation_setup(n_days)

    # Insulin setup -- basal rate
    basal_rate = BR * 1000 / t_end

    # Total insulin rate (adding bolus rate)
    insulin_rate = basal_rate + (meals * IC)

    # Running the simulation
    blood_glucose_level = [X0[4]]

    for i in times[1:]:

        # Updating parameters
        integrator.set_f_params(insulin_rate[i], meals[i], P)

        # Integrating the ODE
        integrator.integrate(integrator.t+dt)

        # Appending the data TODO: preallocation?
        blood_glucose_level.append(integrator.y[4])

    # Converting to mg/dl
    blood_glucose_level = np.asarray(blood_glucose_level) * 18 / P[12]

    return blood_glucose_level

def calculate_reward(blood_glucose_level):
    """
    Positive reward if within normal glycemic range, zero otherwise. If reward_flag is zero then De Paula's method is used
    """

    reward_flag = 1
    # reward_flag = 1

    if reward_flag == 1:
        ''' Binary reward function'''
        low_bg = 70
        high_bg = 120

        if np.max(blood_glucose_level) < high_bg and np.min(blood_glucose_level) > low_bg:
            reward = 0
        else:
            reward = -1

    elif reward_flag == 2:
        ''' Squared cost function '''
        bg_ref = 90

        reward = - (blood_glucose_level - bg_ref)**2

    elif reward_flag == 3:
        ''' Absolute cost function '''
        bg_ref = 90

        reward = - abs(blood_glucose_level - bg_ref)

    # elif reward_flag == 4:
        # ''' Squared cost with insulin constraint '''
        # bg_ref = 80

        # reward = - (blood_glucose_level - bg_ref)**2 - 
    else:
        ''' Gaussian reward function '''
        bg_ref = 90
        h = 30

        # reward =  10 * np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 /h**2) - 5
        reward =  np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 /h**2)


    return reward

def hovorka_parameters(BW):
    """
    PATIENT PARAMETERS
    BW - body weight in kilos
    """

    # Patient-dependent parameters:
    V_I = 0.12*BW              # Insulin volume [L]
    V_G = 0.16*BW              # Glucose volume [L]
    F_01 = 0.0097*BW           # Non-insulin-dependent glucose flux [mmol/min]
    EGP_0 = 0.0161*BW          # EGP extrapolated to zero insulin concentration [mmol/min]

    # Patient-independent(?) parameters:
    S_IT = 51.2e-4             # Insulin sensitivity of distribution/transport [L/min*mU]
    S_ID = 8.2e-4              # Insulin sensitivity of disposal [L/min*mU]
    S_IE = 520e-4              # Insluin sensitivity of EGP [L/mU]
    tau_G = 40                 # Time-to-maximum CHO absorption [min]
    tau_I = 55                 # Time-to-maximum of absorption of s.c. injected short-acting insulin [min]
    A_G = 0.8                  # CHO bioavailability [1]
    k_12 = 0.066               # Transfer rate [min]
    k_a1 = 0.006               # Deactivation rate of insulin on distribution/transport [1/min]
    k_b1 = S_IT*k_a1           # Activation rate of insulin on distribution/transport
    k_a2 = 0.06                # Deactivation rate of insulin on dsiposal [1/min]
    k_b2 = S_ID*k_a2           # Activation rate of insulin on disposal
    k_a3 = 0.03                # Deactivation rate of insulin on EGP [1/min]
    k_b3 = S_IE*k_a3           # Activation rate of insulin on EGP
    k_e = 0.138                # Insluin elimination from Plasma [1/min]

    # Summary of the patient's values:
    P = [tau_G, tau_I, A_G, k_12, k_a1, k_b1, k_a2, k_b2, k_a3, k_b3, k_e, V_I, V_G, F_01, EGP_0]

    return P

def hovorka_model(t, x, u, D, P): ## This is the ode version
    """HOVORKA DIFFERENTIAL EQUATIONS
    # t:    Time window for the simulation. Format: [t0 t1], or [t1 t2 t3 ... tn]. [min]
    # x:    Initial conditions
    # u:    Amount of insulin insulin injected [mU/min]
    # D:    CHO eating rate [mmol/min]
    # P:    Model fixed parameters
    #
    # Syntax :
    # [T, X] = ode15s(@Hovorka, [t0 t1], xInitial0, odeOptions, u, D, p);
    """
    # TODO: update syntax in docstring
    import numpy as np

    # u, D, P = args

    # Defining the various equation names
    D1 = x[ 0 ]               # Amount of glucose in compartment 1 [mmol]
    D2 = x[ 1 ]               # Amount of glucose in compartment 2 [mmol]
    S1 = x[ 2 ]               # Amount of insulin in compartment 1 [mU]
    S2 = x[ 3 ]               # Amount of insulin in compartment 2 [mU]
    Q1 = x[ 4 ]               # Amount of glucose in the main blood stream [mmol]
    Q2 = x[ 5 ]               # Amount of glucose in peripheral tissues [mmol]
    I =  x[ 6 ]                # Plasma insulin concentration [mU/L]
    x1 = x[ 7 ]               # Insluin in muscle tissues [1], x1*Q1 = Insulin dependent uptake of glucose in muscles
    x2 = x[ 8 ]               # [1], x2*Q2 = Insulin dependent disposal of glucose in the muscle cells
    x3 = x[ 9 ]              # Insulin in the liver [1], EGP_0*(1-x3) = Endogenous release of glucose by the liver

    # Unpack data
    tau_G = P[ 0 ]               # Time-to-glucose absorption [min]
    tau_I = P[ 1 ]               # Time-to-insulin absorption [min]
    A_G = P[ 2 ]                 # Factor describing utilization of CHO to glucose [1]
    k_12 = P[ 3 ]                # [1/min] k_12*Q2 = Transfer of glucose from peripheral tissues (ex. muscle to the blood)
    k_a1 = P[ 4 ]                # Deactivation rate [1/min]
    k_b1 = P[ 5 ]                # [L/(mU*min)]
    k_a2 = P[ 6 ]                # Deactivation rate [1/min]
    k_b2 = P[ 7 ]                # [L/(mU*min)]
    k_a3 = P[ 8 ]                # Deactivation rate [1/min]
    k_b3 = P[ 9 ]               # [L/(mU*min)]
    k_e = P[ 10 ]                # Insulin elimination rate [1/min]
    V_I = P[ 11 ]                # Insulin distribution volume [L]
    V_G = P[ 12 ]                # Glucose distribution volume [L]
    F_01 = P[ 13 ]               # Glucose consumption by the central nervous system [mmol/min]
    EGP_0 = P[ 14 ]              # Liver glucose production rate [mmol/min]

    # Certain parameters are defined
    U_G = D2/tau_G             # Glucose absorption rate [mmol/min]
    U_I = S2/tau_I             # Insulin absorption rate [mU/min]

    # Constitutive equations
    G = Q1/V_G                 # Glucose concentration [mmol/L]

    if (G>=4.5):
        F_01c = F_01           # Consumption of glucose by the central nervous system [mmol/min
    else:
        F_01c = F_01*G/4.5     # Consumption of glucose by the central nervous system [mmol/min]

    if (G>=9):
        F_R = 0.003*(G-9)*V_G  # Renal excretion of glucose in the kidneys [mmol/min]
    else:
        F_R = 0                # Renal excretion of glucose in the kidneys [mmol/min]

    # Mass balances/differential equations
    xdot = np.zeros (10);

    xdot[ 0 ] = A_G*D-D1/tau_G                                # dD1
    xdot[ 1 ] = D1/tau_G-U_G                                  # dD2
    xdot[ 2 ] = u-S1/tau_I                                    # dS1
    xdot[ 3 ] = S1/tau_I-U_I                                  # dS2
    xdot[ 4 ] = -(F_01c+F_R)-x1*Q1+k_12*Q2+U_G+EGP_0*(1-x3)   # dQ1
    xdot[ 5 ] = x1*Q1-(k_12+x2)*Q2                            # dQ2
    xdot[ 6 ] = U_I/V_I-k_e*I                                 # dI
    xdot[ 7 ] = k_b1*I-k_a1*x1                                # dx1
    xdot[ 8 ] = k_b2*I-k_a2*x2                                # dx2
    xdot[ 9 ] = k_b3*I-k_a3*x3                               # dx3

    return xdot

def hovorka_model_tuple(x, *pars):
    """HOVORKA DIFFERENTIAL EQUATIONS without time variable
    # t:    Time window for the simulation. Format: [t0 t1], or [t1 t2 t3 ... tn]. [min]
    # x:    Initial conditions
    # u:    Amount of insulin insulin injected [mU/min]
    # D:    CHO eating rate [mmol/min]
    # P:    Model fixed parameters
    #
    """
    # TODO: update syntax in docstring
    import numpy as np

    # Unpacking_parameters
    u, D, P = pars


    # Defining the various equation names
    D1 = x[ 0 ]               # Amount of glucose in compartment 1 [mmol]
    D2 = x[ 1 ]               # Amount of glucose in compartment 2 [mmol]
    S1 = x[ 2 ]               # Amount of insulin in compartment 1 [mU]
    S2 = x[ 3 ]               # Amount of insulin in compartment 2 [mU]
    Q1 = x[ 4 ]               # Amount of glucose in the main blood stream [mmol]
    Q2 = x[ 5 ]               # Amount of glucose in peripheral tissues [mmol]
    I =  x[ 6 ]                # Plasma insulin concentration [mU/L]
    x1 = x[ 7 ]               # Insluin in muscle tissues [1], x1*Q1 = Insulin dependent uptake of glucose in muscles
    x2 = x[ 8 ]               # [1], x2*Q2 = Insulin dependent disposal of glucose in the muscle cells
    x3 = x[ 9 ]              # Insulin in the liver [1], EGP_0*(1-x3) = Endogenous release of glucose by the liver

    # Unpack data
    tau_G = P[ 0 ]               # Time-to-glucose absorption [min]
    tau_I = P[ 1 ]               # Time-to-insulin absorption [min]
    A_G = P[ 2 ]                 # Factor describing utilization of CHO to glucose [1]
    k_12 = P[ 3 ]                # [1/min] k_12*Q2 = Transfer of glucose from peripheral tissues (ex. muscle to the blood)
    k_a1 = P[ 4 ]                # Deactivation rate [1/min]
    k_b1 = P[ 5 ]                # [L/(mU*min)]
    k_a2 = P[ 6 ]                # Deactivation rate [1/min]
    k_b2 = P[ 7 ]                # [L/(mU*min)]
    k_a3 = P[ 8 ]                # Deactivation rate [1/min]
    k_b3 = P[ 9 ]               # [L/(mU*min)]
    k_e = P[ 10 ]                # Insulin elimination rate [1/min]
    V_I = P[ 11 ]                # Insulin distribution volume [L]
    V_G = P[ 12 ]                # Glucose distribution volume [L]
    F_01 = P[ 13 ]               # Glucose consumption by the central nervous system [mmol/min]
    EGP_0 = P[ 14 ]              # Liver glucose production rate [mmol/min]

    # Certain parameters are defined
    U_G = D2/tau_G             # Glucose absorption rate [mmol/min]
    U_I = S2/tau_I             # Insulin absorption rate [mU/min]

    # Constitutive equations
    G = Q1/V_G                 # Glucose concentration [mmol/L]

    if (G>=4.5):
        F_01c = F_01           # Consumption of glucose by the central nervous system [mmol/min
    else:
        F_01c = F_01*G/4.5     # Consumption of glucose by the central nervous system [mmol/min]

    if (G>=9):
        F_R = 0.003*(G-9)*V_G  # Renal excretion of glucose in the kidneys [mmol/min]
    else:
        F_R = 0                # Renal excretion of glucose in the kidneys [mmol/min]

    # Mass balances/differential equations
    xdot = np.zeros (10);

    xdot[ 0 ] = A_G*D-D1/tau_G                                # dD1
    xdot[ 1 ] = D1/tau_G-U_G                                  # dD2
    xdot[ 2 ] = u-S1/tau_I                                    # dS1
    xdot[ 3 ] = S1/tau_I-U_I                                  # dS2
    xdot[ 4 ] = -(F_01c+F_R)-x1*Q1+k_12*Q2+U_G+EGP_0*(1-x3)   # dQ1
    xdot[ 5 ] = x1*Q1-(k_12+x2)*Q2                            # dQ2
    xdot[ 6 ] = U_I/V_I-k_e*I                                 # dI
    xdot[ 7 ] = k_b1*I-k_a1*x1                                # dx1
    xdot[ 8 ] = k_b2*I-k_a2*x2                                # dx2
    xdot[ 9 ] = k_b3*I-k_a3*x3                               # dx3

    return xdot

def bgplot(blood_glucose_level):
    """
    Plotting the blood glucose curve
    """
    bg_low = 70 *np.ones(len(blood_glucose_level))
    bg_high = 170 *np.ones(len(blood_glucose_level))
    # n_days = int(len(blood_glucose_level)/1440)

    l = int(len(blood_glucose_level))

    fig, ax = plt.subplots()
    ax.plot(blood_glucose_level)
    # ax.fill_between(range(0, t_end*n_days), bg_low, bg_high, alpha=.3, facecolor='green')
    ax.fill_between(range(0, l), bg_low, bg_high, alpha=.3, facecolor='green')
    plt.title('Blood glucose curve')
    plt.ion()
    plt.show(block=True)


if __name__ == '__main__':
    # """ Main method testing the simulator """

    # TODO: This is no longer in use!
    # Insulin parameters
    # basal = 8.3 # units per day
    # bolus = 8.8 # [mU/mmol]

    # n_days = 10
    # # Running simulation
    # blood_glucose_level = simulate(bolus, basal, n_days)

    # # Calculating reward
    # r = calculate_reward(blood_glucose_level)
    # print('Reward is:', r)

    # # Plotting
    # bgplot(blood_glucose_level)
    print('Hello')

