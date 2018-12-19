import gym
import time
import numpy as np

env1 = gym.make('HovorkaGaussian-v0')
env2 = gym.make('CambridgeGaussian-v0')

def time_env(env):
    t1 = time.time()
    env.reset()
    for i in range(48):
        env.step(np.array([0]))

    t2 = time.time()

    return t2-t1


time_hovorka = time_env(env2)
time_cambridge = time_env(env2)
