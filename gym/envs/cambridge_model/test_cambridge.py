import numpy as np
import gym
# import seaborn as sns
# sns.set()

from pylab import plot, figure, title, show, ion, legend
from gym.envs.cambridge_model.subject import subject

# env = gym.make('Cambridge-v0')
# env = gym.make('CambridgeGaussian-v0')
# env = gym.make('CambridgeGaussianInsulin-v0')
# env = gym.make('CambridgeBinary-v0')

env = gym.make('CambridgeAbsolute-v0')
# env.env.P = subject(1)

# basal = 0
# env.env.reset_basal_manually = 6.43

env.reset()
reward = []

# Simulation states
bg = []
cgm = []

for i in range(48):

    # Step for the minimal/hovorka model
    s, r, d, i = env.step(np.array([-1]))
    reward.append(r)
    # print(r)

    # Step for the discrete Hovorka incremental
    # env.step(2)
    bg.append(env.env.simulation_state[4])
    cgm.append(env.env.simulation_state[-1] * 18)



# env.render()
# plot(env.env.bg_history)
figure()
plot(bg)
plot(cgm)
legend(['bg', 'cgm'])
title('bg and cgm')
# ion()
show()

