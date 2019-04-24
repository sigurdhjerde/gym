"""
PATIENT PARAMETERS
BW - body weight in kilos
"""
from numpy.random import normal as normal_dist
from numpy.random import uniform as uniform_dist
import numpy as np


def sample_patient():


    # Body weight
    # BW = normal_dist(74.9, 14.4)
    BW = uniform_dist(45, 95)

    # ========================================
    # Clinically derived parameters -- FIXED
    # ========================================

    F_01  = 11.1e-3 * BW         # Noninsulin-dependent glucose flux (μmol kg-1 min-1)
    EGP_0 = 16.9e-3 * BW         # Endogenous glucose production extrapolated to zero insulin concentration (μmol kg -1 min-1 )
    k_12  = 0.060        # Transfer rate from nonaccessible to accessible glucose compartment (min-1)

    S_ID = 5.05e-4      # Insulin sensitivity of glucose disposal (min-1 per mU liter-1)
    S_IE  = 0.019        # Insulin sensitivity of suppression of endogenous glucose production (per mU liter -1)
    S_IT  = 18.41e-4     # Insulin sensitivity of glucose transport/distribution (min-1 per mU liter -1)

    k_a1  = 0.0034       # Activation rate of remote insulin effect on glucose distribution (min-1)
    k_a2  = 0.056        # Activation rate of remote insulin effect on glucose disposal (min -1)
    k_a3  = 0.024        # Activation rate of remote insulin effect on endogenous glucose production (min-1 )

    k_b1 = S_IT * k_a1
    k_b2 = S_ID * k_a1
    k_b3 = S_IE * k_a1

    # ==================================================
    # Fixed parameters from original Hovorka model
    # ==================================================

    # F_01 = 0.0097*BW           # Non-insulin-dependent glucose flux [mmol/min]
    # EGP_0 = 0.0161*BW          # EGP extrapolated to zero insulin concentration [mmol/min]
    # k_12 = 0.066               # Transfer rate [min]

    # S_IT = 51.2e-4             # Insulin sensitivity of distribution/transport [L/min*mU]
    # S_ID = 8.2e-4              # Insulin sensitivity of disposal [L/min*mU]
    # S_IE = 520e-4              # Insluin sensitivity of EGP [L/mU]

    # k_a1 = 0.006               # Deactivation rate of insulin on distribution/transport [1/min]
    # k_b1 = S_IT*k_a1           # Activation rate of insulin on distribution/transport
    # k_a2 = 0.06                # Deactivation rate of insulin on dsiposal [1/min]
    # k_b2 = S_ID*k_a2           # Activation rate of insulin on disposal
    # k_a3 = 0.03                # Deactivation rate of insulin on EGP [1/min]
    # k_b3 = S_IE*k_a3           # Activation rate of insulin on EGP

    # =======================================
    # Parameters sampled from distributions
    # =======================================


    # V_G         = np.clip(np.exp(normal_dist(np.log(0.15), 0.23)), 0.09, 0.25)      # Glucose distribution volume (liter kg -1)
    # V_G         = V_G * BW
    # R_thr       = np.clip(normal_dist(9, 1.5), 7.5, 15)               # Renal clearance threshold (mmol liter -1)
    # R_cl        = np.clip(normal_dist(0.01, 0.025), 0.003, 0.03)          # Renal clearance rate (min-1 )
    # V_I         = np.clip(normal_dist(0.12, 0.012),0.08, 0.18)         # Insulin distribution volume (liter kg-1)
    # V_I         = V_I * BW
    # tau_I       = 1 / np.clip(normal_dist(0.018, 0.0045), 0.005, 0.06)        # Insulin absorption rate (min -1)
    # k_e         = np.clip(normal_dist(0.14, 0.0345), 0.05, 0.30)          # Insulin elimination rate (min-1)
    # A_G         = uniform_dist(70, 120) / 100              # Bioavailability of CHO (%)
    # tau_G       = 1 / np.clip(np.exp(normal_dist(-3.689, 0.25)), 0.02, .035)    # Time-to-maximum of CHO absorption (min)
    # ka_int      = np.exp(normal_dist(-2.372, 1.092))    # Transfer-rate constant between interstitial and plasma glucose compartment (min-1)

    # =======================================================
    # Squared and not-clipped version of the parameters
    # =======================================================

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

    # Summary of the patient's values:
    P = [tau_G, tau_I, A_G, k_12, k_a1, k_b1, k_a2, k_b2, k_a3, k_b3, k_e, V_I, V_G, F_01, EGP_0, ka_int, R_cl, R_thr]

    return P, BW
