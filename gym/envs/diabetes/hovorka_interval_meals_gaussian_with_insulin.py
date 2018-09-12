from gym.envs.diabetes import hovorka_base

class HovorkaGaussianInsulin(hovorka_base.HovorkaBase):

    def _update_parameters(self):
        ''' Update parameters of model,
        this is only used for inherited classes'''

        meal_times = [480,600,720,960,1080,1920,2040,2160,2400,2520]
        meal_amounts = [70,15,60,15,65,70,15,60,15,65]
        reward_flag = 'gaussian_with_insulin'

        # Initialization flag: 'random' or 'fixed'
        bg_init_flag = 'random'
        max_insulin_action = 300

        return meal_times, meal_amounts, reward_flag, bg_init_flag, max_insulin_action
