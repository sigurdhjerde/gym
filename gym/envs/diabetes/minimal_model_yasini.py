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

    # Parameters Normal
    p1 = 0.0317
    p2 = 0.0123
    p3 = 4.92e-6
    n = 0.2659
    # gamma = 0.00039
    gamma = 0.00001
    h = 79.0353

    # Parameters Patient 1
    # p1 = 0.03
    # p2 = 0.0107
    # p3 = 5.3e-6
    # n = 0.264
    # # gamma = 0.0042
    # gamma = 0
    # h = 80.25

    # Parameters patient 2
    # p1 = 0
    # p2 = 0.0072
    # p3 = 2.16e-6
    # n = 0.2465
    # gamma = 0.0039
    # # gamma = 0
    # h = 77.5783

    # Parameters patient 4
    # p2 = 0.0316
    # p2 = 0.0107
    # p3 = 5.3e-6
    # n = 0.2640
    # gamma = 0.0042
    # h = 80.2576

    # Giovanni patient
    # p1 = 0.000317
    # p2 = 0.0123
    # p3 = 4.92e-6
    # n = 0.2569
    # gamma = .0039 * 0.5
    # # gamma = 0
    # h = 79.0353


    # From Hovorka parameters -- Phuong used these in his experiments
    # p1 = 0.02; p2 = 0.028; p3 = 1e-4

    # Reference basal values for BG and insulin
    G_b = 80
    I_b = 7

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
