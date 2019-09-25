import gym

from gym.envs.registration import register
import numpy as np

from scipy.io import loadmat

# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def matlab_to_python(patient_num):
    '''
    Converting matlab parameters to python parameters
    (only changing the orders)

              'MCHO': 0
              'TauS': 1
              'GBasal': 2
              'w': 3
              'EGP0': 4
              'F01': 5
              'k12': 6
              'ka1': 7
              'ka2': 8
              'ka3': 9
              'St': 10
              'Sd': 11
              'Se': 12
              'ka': 13
              'ke': 14
              'Vi': 15
              'Vg': 16
              'Ip0': 17
              'Ub': 18
              'Bio': 19
              'TauM': 20
              'carbF': 21
              'ISF': 22
              'Diet': 23
              'TDD': 24
    '''
    # patients_all = loadmat('/home/miguel/git/pg_diabetes/garage/anas_patients/patientAdultMcGill.mat')
    patients_all = loadmat('/home/jonas/Documents/git/pg_diabetes/garage/anas_patients/patientAdultMcGill.mat')
    params_all = patients_all['param']

    params = params_all[0][patient_num]
    # print(params)

    # Patient-dependent parameters:
    V_I = params[15]/1000 * params[3]              # Insulin volume [L]
    V_G = params[16]/1000 * params[3]              # Glucose volume [L]
    F_01 = params[5]/1000 * params[3]           # Non-insulin-dependent glucose flux [mmol/min]
    EGP_0 = params[4]/1000 *params[3]          # EGP extrapolated to zero insulin concentration [mmol/min]

    # Patient-independent(?) parameters:
    S_IT = params[10]          # Insulin sensitivity of distribution/transport [L/min*mU]
    S_ID = params[11]              # Insulin sensitivity of disposal [L/min*mU]
    S_IE = params[12]              # Insluin sensitivity of EGP [L/mU]

    tau_G = params[20]                 # Time-to-maximum CHO absorption [min]
    tau_I = 1/params[13]                 # Time-to-maximum of absorption of s.c. injected short-acting insulin [min]

    A_G = params[19]                  # CHO bioavailability [1]
    k_12 = params[6]               # Transfer rate [min]

    k_a1 = params[7]               # Deactivation rate of insulin on distribution/transport [1/min]
    k_b1 = S_IT*k_a1           # Activation rate of insulin on distribution/transport
    k_a2 = params[8]           # Deactivation rate of insulin on dsiposal [1/min]
    k_b2 = S_ID*k_a2           # Activation rate of insulin on disposal
    k_a3 = params[9]                # Deactivation rate of insulin on EGP [1/min]
    k_b3 = S_IE*k_a3           # Activation rate of insulin on EGP

    k_e = params[14]                # Insulin elimination from Plasma [1/min]

    k_a = 1/params[1]
    R_cl = 0.01
    R_thr = 11

    # Summary of the patient's values:
    P = [tau_G, tau_I, A_G, k_12, k_a1, k_b1, k_a2, k_b2, k_a3, k_b3, k_e, V_I, V_G, F_01, EGP_0, k_a, R_cl, R_thr]

    # basal insulin rate
    basal_rate = params[18]/60 *1000

    tdd = params[24]

    carb_factor = params[21]

    return P, basal_rate.flatten(), carb_factor.flatten(), tdd




if __name__=='__main__':
    '''Loading the stable patients shared by Anas
    '''

    patients_all = loadmat('/home/jonas/Documents/git/EXTERNAL/artificial-pancreas-simulator-uit/library/virtual-patients/@HovorkaPatientUIT/patientAdultMcGill.mat')
    params = patients_all['param']

    # Testing a single patient

    env = gym.make('HovorkaCambridge-v0')

    plt.figure()
    for i in range(100):
        logger.info('Patient number ' + str(i))
        P, basal_rate, carb_factor, _ = matlab_to_python(params[0][i])
        env.env.P = P
        env.env.reset_basal_manually = basal_rate
        env.env.bolus = carb_factor

        env.reset()
    
        for i in range(48):
            env.step(basal_rate)

        plt.plot(env.env.bg_history)
        
    plt.fill_between(range(1440), 70, 180, alpha=.20)
    plt.show(block=False)
