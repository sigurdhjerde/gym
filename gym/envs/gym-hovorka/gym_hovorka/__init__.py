from gym.envs.registration import register

register(
        id='hovorka-v0',
        entry_point='gym_hovorka.envs.hovorka:HovorkaDiabetes',
)


