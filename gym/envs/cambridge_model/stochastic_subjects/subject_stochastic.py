"""
PATIENT PARAMETERS
BW - body weight in kilos
"""
from numpy.random import normal as normal_dist
from numpy.random import uniform as uniform_dist
import numpy as np


def sample_patient():

    # =======================================================
    # Parameters from Boiroux' 2018 paper
    # =======================================================

    # Body weight
    BW = uniform_dist(65, 95)

    F_01  = normal_dist(9.7e-3, 0.0022**2) * BW         # Noninsulin-dependent glucose flux (μmol kg-1 min-1)
    EGP_0 = normal_dist(16.1e-3,0.0039**2) * BW         # Endogenous glucose production extrapolated to zero insulin concentration (μmol kg -1 min-1 )
    k_12  = normal_dist(0.0649, 0.0282**2)        # Transfer rate from nonaccessible to accessible glucose compartment (min-1)

    S_ID = normal_dist(8.2e-4, (7.84E-4)**2)      # Insulin sensitivity of glucose disposal (min-1 per mU liter-1)
    S_IE  = normal_dist(520e-4, (306.2e-4)**2)        # Insulin sensitivity of suppression of endogenous glucose production (per mU liter -1)
    S_IT  = normal_dist(51.2e-4, (32.09e-4)**2)     # Insulin sensitivity of glucose transport/distribution (min-1 per mU liter -1)

    k_a1  = normal_dist(0.0055, 0.0056**2)       # Activation rate of remote insulin effect on glucose distribution (min-1)
    k_a2  = normal_dist(0.0683, 0.0507**2)        # Activation rate of remote insulin effect on glucose disposal (min -1)
    k_a3  = normal_dist(0.0304, 0.0235**2)        # Activation rate of remote insulin effect on endogenous glucose production (min-1 )

    k_b1 = S_IT * k_a1
    k_b2 = S_ID * k_a1
    k_b3 = S_IE * k_a1

    V_G         = np.exp(normal_dist(np.log(0.15), 0.23**2))      # Glucose distribution volume (liter kg -1)
    V_G         = V_G * BW
    R_thr       = normal_dist(9, 1.5**2)               # Renal clearance threshold (mmol liter -1)
    R_cl        = normal_dist(0.01, 0.025**2)          # Renal clearance rate (min-1 )
    V_I         = normal_dist(0.12, 0.012**2)         # Insulin distribution volume (liter kg-1)
    V_I         = V_I * BW
    tau_I       = 1 / normal_dist(0.018, 0.0045**2)        # Insulin absorption rate (min -1)
    k_e         = normal_dist(0.14, 0.035**2)          # Insulin elimination rate (min-1)
    A_G         = uniform_dist(70, 120) / 100              # Bioavailability of CHO (%)
    tau_G       = 1 / np.exp(normal_dist(-3.689, 0.25**2)) # Time-to-maximum of CHO absorption (min)
    ka_int      = np.exp(normal_dist(-2.372, 1.092**2))    # Transfer-rate constant between interstitial and plasma glucose compartment (min-1)

    # =======================================================
    # Non - squared version of the parameters
    # =======================================================

    # F_01  = normal_dist(9.7e-3, 0.0022) * BW         # Noninsulin-dependent glucose flux (μmol kg-1 min-1)
    # EGP_0 = normal_dist(16.1e-3,0.0039) * BW         # Endogenous glucose production extrapolated to zero insulin concentration (μmol kg -1 min-1 )
    # k_12  = normal_dist(0.0649, 0.0282)        # Transfer rate from nonaccessible to accessible glucose compartment (min-1)
# 
    # S_ID = normal_dist(8.2e-4, (7.84E-4))      # Insulin sensitivity of glucose disposal (min-1 per mU liter-1)
    # S_IE  = normal_dist(520e-4, (306.2e-4))        # Insulin sensitivity of suppression of endogenous glucose production (per mU liter -1)
    # S_IT  = normal_dist(51.2e-4, (32.09e-4))     # Insulin sensitivity of glucose transport/distribution (min-1 per mU liter -1)
# 
    # k_a1  = normal_dist(0.0055, 0.0056)       # Activation rate of remote insulin effect on glucose distribution (min-1)
    # k_a2  = normal_dist(0.0683, 0.0507)        # Activation rate of remote insulin effect on glucose disposal (min -1)
    # k_a3  = normal_dist(0.0304, 0.0235)        # Activation rate of remote insulin effect on endogenous glucose production (min-1 )
# 
    # k_b1 = S_IT * k_a1
    # k_b2 = S_ID * k_a1
    # k_b3 = S_IE * k_a1
# 
    # V_G         = np.exp(normal_dist(np.log(0.15), 0.23))      # Glucose distribution volume (liter kg -1)
    # V_G         = V_G * BW
    # R_thr       = normal_dist(9, 1.5)               # Renal clearance threshold (mmol liter -1)
    # R_cl        = normal_dist(0.01, 0.025)          # Renal clearance rate (min-1 )
    # V_I         = normal_dist(0.12, 0.012)         # Insulin distribution volume (liter kg-1)
    # V_I         = V_I * BW
    # tau_I       = 1 / normal_dist(0.018, 0.0045)        # Insulin absorption rate (min -1)
    # k_e         = normal_dist(0.14, 0.035)          # Insulin elimination rate (min-1)
    # A_G         = uniform_dist(70, 120) / 100              # Bioavailability of CHO (%)
    # tau_G       = 1 / np.exp(normal_dist(-3.689, 0.25)) # Time-to-maximum of CHO absorption (min)
    # ka_int      = np.exp(normal_dist(-2.372, 1.092))    # Transfer-rate constant between interstitial and plasma glucose compartment (min-1)

    # Summary of the patient's values:
    P = [tau_G, tau_I, A_G, k_12, k_a1, k_b1, k_a2, k_b2, k_a3, k_b3, k_e, V_I, V_G, F_01, EGP_0, ka_int, R_cl, R_thr]

    return P, BW
