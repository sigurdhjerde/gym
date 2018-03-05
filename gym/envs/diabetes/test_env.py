import numpy as np
import gym
import seaborn as sns
sns.set()

from pylab import plot, figure, title, show

env = gym.make('HovorkaDiabetes-v0')
# env = gym.make('MinimalDiabetes-v0')
# env = gym.make('HovorkaDiabetesIncremental-v0')


basal = 6.66
env.env.init_basal = basal

env.reset()

for i in range(100):

    # Step for the minimal/hovorka model
    env.step(np.array([basal]))

    # Step for the discrete Hovorka incremental
    # env.step(2)


# env.render()
figure()
plot(env.env.bg_history)
show()

