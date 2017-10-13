# from gym.envs.diabetes.hovorka import HovorkaDiabetes

from gym.envs.registration import register

register(
        id = 'HovorkaDiabetes-v0',
        entry_point = 'diabetes.hovorka:HovorkaDiabetes',
        timestep_limit = 1440,
        max_episode_steps=1440
        )
register(
        id = 'MinimalDiabetes-v0',
        entry_point = 'diabetes.minimal_env:MinimalDiabetes',
        timestep_limit = 1440,
        max_episode_steps = 1440
        )
register(
        id = 'MinimalDiabetesMeals-v0',
        entry_point = 'diabetes.minimal_env_meals:MinimalDiabetesMeals',
        timestep_limit = 1440,
        max_episode_steps = 1440
        )
