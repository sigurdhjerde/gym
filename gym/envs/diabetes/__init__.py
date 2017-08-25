# from gym.envs.diabetes.hovorka import HovorkaDiabetes

from gym.envs.registration import register

register(
        id = 'HovorkaDiabetes-v0',
        entry_point = 'diabetes.hovorka:HovorkaDiabetes'
        )
