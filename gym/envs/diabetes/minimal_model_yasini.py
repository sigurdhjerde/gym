import numpy as np

def carb_model(t, xk, D, P):
    ''' Copied from (matlab) Phuong Ngo - UiT 04/2017

    Model adopted from the Hovorka model
    '''
    D = D/180*1000;

    tau_G = P[0]               # Time-to-glucose absorption [min]
    A_G = P[2]                 # Factor describing utilization of CHO to glucose
    D1 = xk[0]
    D2 = xk[1]
    # dummy_var = xk[2]

    # Converting to the same style as the Hovorka model
    xdot = np.zeros([3,1])
    xdot[0] = A_G*D - D1/tau_G                                # dD1
    xdot[1] = D1/tau_G - D2/tau_G                                  # dD2

    U_G = xdot[1]/tau_G             # Glucose absorption rate [mmol/min]
    xdot[2] = U_G

    # return xdot, U_G
    return xdot


def insulin_model(t, xk, uk, U_G, P):
# def insulin_model(t, xk, uk, P):
    '''
    The insulin glucose model from the Yasini paper, one extra eq compared to
    what Phuong was using.

    '''

    # Reference basal values for BG and insulin
    G_b = 90
    I_b = 7

    # Parameters Normal
    # p1 = 0.1
    # p2 = 0.01
    # p3 = 5e-6
    # n = 0.3
    # gamma = 0.002
    # h = G_b

    # Parameters diabetes
    p1 = 0.0001
    p2 = 0.01
    p3 = 5e-6
    n = 0.3
    gamma = 0.00002
    h = G_b

    # tau_G = P[1] # Time-to-glucose absorption [min]
    V_G = P[12] # Glucose Volume Distribution (L)

    # Unpacking input variables
    G = xk[0]           # Glucose
    X = xk[1]           # Remote insulin
    I = xk[2]           # Plasma insulin
    D = U_G / V_G * 18  # Carbohydrates


    # Equations
    dg_dt = -p1*(G - G_b) - X*G + D

    dx_dt= -p2*X + p3*(I - I_b)

    di_dt = -n*(I - I_b) + gamma * max((G - h), 0)*t + uk
    # di_dt = -n*(I - I_b) + gamma * max((G - h), 0) + uk


    # Output
    xdot = np.zeros([3,1])

    xdot[0] = dg_dt
    xdot[1] = dx_dt
    xdot[2] = di_dt


    return xdot
