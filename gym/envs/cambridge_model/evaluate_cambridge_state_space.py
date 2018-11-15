import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show, ion, legend
from gym.envs.cambridge_model.subject import subject

# env = gym.make('Cambridge-v0')
# env = gym.make('CambridgeGaussian-v0')
# env = gym.make('CambridgeGaussianInsulin-v0')
# env = gym.make('CambridgeBinary-v0')

env = gym.make('CambridgeAbsolute-v0')


# Simulation states

def run_episode(env, action, sub, init_basal):

    bg = []
    cgm = []
    reward = []

    env.env.P = subject(sub)
    env.env.reset_basal_manually = init_basal
    env.reset()

    for i in range(48):

        # Step for the minimal/hovorka model
        s, r, d, i = env.step(np.array([action]))

        reward.append(r)
        bg.append(env.env.simulation_state[4])
        cgm.append(env.env.simulation_state[-1] * env.env.P[12])

    return bg, reward, cgm

# running through all subjects with different insulin action
insulin_rates = np.linspace(0, 15, 16)


for i in range(6):
    env.env.P = subject(i+1)
    print('Reset values for patient '  + str(i))
    for j in range(len(insulin_rates)):
        env.env.reset_basal_manually = insulin_rates[j]
        s = env.reset()
        print(s[0])

