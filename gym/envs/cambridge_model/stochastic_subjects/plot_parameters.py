import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim, stem, subplot, xlim, suptitle
from gym.envs.cambridge_model.subject_stochastic import sample_patient
ion()

env = gym.make('CambridgeAbsolute-v0')
env2 = gym.make('CambridgeAbsolute-v0')
Pdef = env2.env.P

env.env.reset_basal_manually = 6.43

# pars = np.load('parameters_hovorka_fixed.npy')
pars = np.load('parameters_cambridge.npy')
# pars = pars[:, 0:29]
bw = np.load('parameters_cambridge_bw.npy')

palette = sns.color_palette('Blues', 150)
# Plotting episodes for all sampled patients
figure()
for i in range(30):
    env.env.P = pars[:, i]
    env.reset()

    for j in range(40):
        s, r, d, _ = env.step(np.array([6.43]))

    plot(range(len(env.env.bg_history)), env.env.bg_history, color=palette[int(np.round(bw[i]))])
show()

# List of all parameters
title_string = ['tau_G', 'tau_I', 'A_G', 'k_12', 'k_a1', 'k_b1', 'k_a2', 'k_b2', 'k_a3', 'k_b3', 'k_e', 'V_I', 'V_G', 'F_01', 'EGP_0', 'ka_int', 'R_cl', 'R_thr']

# =======================
# Stochastic parameters
# =======================


figure()

# Height of KDE, found from plots!
kde_height = [0.04, 0.025, 2.0, 10, 0.2, 0.150, 4, 25 ,0.25]

count = 1
for i in [0, 1, 2, 10, 11 ,12, 15, 16, 17]:
    subplot(3, 3, count)
    sns.kdeplot(pars[i,:], shade=True)
    # stem([Pdef[i],], [np.mean(pars[i,:]),], linefmt='C3', markerfmt='C3o')
    stem([Pdef[i],], [kde_height[count-1],], linefmt='C3', markerfmt='C3o')
    title(title_string[i])
    count += 1


suptitle('Kernel density estimate of all stochastic parameters')
show()

# ==================
# Fixed parameters
# ==================

figure()

index_fixed = [3] + list(range(4, 10)) + [13, 14]

count = 1
for i in index_fixed:
    subplot(3, 3, count)
    stem([Pdef[i]/Pdef[i],], [1,], linefmt='C0', markerfmt='C0o')
    stem([pars[i, 0]/Pdef[i],], [1,], linefmt='C3', markerfmt='C3o')
    xlim(-.2, 1.2)
    count +=1
    title(title_string[i])

suptitle('Fixed parameters compared to original (normalized)')
show()
