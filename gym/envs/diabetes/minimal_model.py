import numpy as np

def carb_model(t, xk, D, P):
    ''' Copied from (matlab) Phuong Ngo - UiT 04/2017 '''
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
    '''Copied from (matlab)Phuong Ngo - UiT 04/2017 '''

    # Parameters
    p1 = 0.02; p2 = 0.028; p3 = 1e-4

    # Reference BG value
    gb = 80

    # tau_G = P[1] # Time-to-glucose absorption [min]
    V_G = P[12] # Glucose Volume Distribution (L)
    xdot = np.zeros([2,1])
    xdot[0] = (-p1*xk[0] - xk[1]*(xk[0] + gb) + U_G/(V_G)*18)
    # xdot[0] = (-p1*xk[0] - xk[1]*(xk[0] + gb) + xk[2]/(V_G)*18)
    xdot[1] = (-p2*xk[1] + p3*uk)

    return xdot
