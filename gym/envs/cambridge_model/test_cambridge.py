import numpy as np
import gym
# import seaborn as sns
# sns.set()

from pylab import plot, figure, title, show, ion, legend, ylim
from gym.envs.cambridge_model.subject_stochastic import sample_patient
from gym.envs.registration import register


pn = 1

register(
    id='cambridge-gaussian-1-v0',
    entry_point='cambridge_gaussian:CambridgeGaussian',
    kwargs={'patient_number': pn}
   )

env = gym.make('cambridge-gaussian-1-v0')
# env = gym.make('Cambridge-v0')
# env = gym.make('CambridgeGaussian-v0')
# env = gym.make('CambridgeGaussianInsulin-v0')
# env = gym.make('CambridgeBinary-v0')
# env = gym.make('CambridgeAbsolute-v0')

env.reset()
reward = []

# Simulation states
bg = []
cgm = []

basal_rates = np.load('init_basal.npy')

for i in range(48):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([basal_rates[pn]]))
    reward.append(r)
    # print(r)

    # Step for the discrete Hovorka incremental
    # env.step(2)
    # bg.append(env.env.simulation_state[4])
    # cgm.append(env.env.simulation_state[-1] * 18)

    bg.append(env.simulation_state[4])
    cgm.append(env.simulation_state[-1] * 18)



# env.render()
# plot(env.env.bg_history)
# figure()
# plot(bg)
plot(cgm)
# ylim(100, 250)
# legend(['bg', 'cgm'])
# title('bg and cgm')
# ion()
show()
