import numpy as np
import gym
import seaborn as sns
sns.set()

env = gym.make('HovorkaDiabetes-v0')

env.reset()

for i in range(1440):
    env.step(np.array([8.3]))


env.render()
