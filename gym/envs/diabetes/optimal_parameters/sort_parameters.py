import numpy as np

# Loading patients
good_patients = np.load('good_patients.npy')
optimal_basal_all = np.load('optimal_basal_min_8.npy')
optimal_bolus_all = np.load('optimal_bolus_all.npy')
pars_all = np.load('parameters_hovorka_new_min_8.npy')
bw_all = np.load('parameters_hovorka_new_min_8_bw.npy')

# Sorting patients
pars = pars_all[:, good_patients]
optimal_basal = optimal_basal_all[good_patients]
optimal_bolus = optimal_bolus_all[good_patients]
bw = bw_all[good_patients]

# Saving
np.save('pars.npy', pars)
np.save('optimal_basal.npy', optimal_basal)
np.save('optimal_bolus.npy', optimal_bolus) 
np.save('bw.npy', bw)
