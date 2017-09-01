# from gym.envs.diabetes.hovorka import HovorkaDiabetes

from gym.envs.registration import register

register(
        id = 'HovorkaDiabetes-v0',
        entry_point = 'diabetes.hovorka:HovorkaDiabetes',
        timestep_limit = 1440,
        max_episode_steps=1440
        )
