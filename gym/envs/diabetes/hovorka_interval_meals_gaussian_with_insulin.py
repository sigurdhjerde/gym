from gym.envs.diabetes import hovorka_base

class HovorkaGaussianInsulin(hovorka_base.HovorkaBase):

    def _update_parameters(self):
        ''' Update parameters of model,
        this is only used for inherited classes'''

<<<<<<< HEAD
        meal_times = [480,600,720,960,1080,1920,2040,2160,2400,2520]
        meal_amounts = [70,15,60,15,65,70,15,60,15,65]
=======
        meal_times = [480, 720, 1080, 1920, 2160, 2520, 3360, 3600, 3960, 4800, 5040, 5400, 6240, 6480, 6840, 7680, 7920, 8280, 9120, 9360, 9720]
        meal_amounts = [50, 70, 70, 50, 70, 70, 50, 70, 70, 50, 70, 70, 50, 70, 70, 50, 70, 70, 50, 70, 70]
>>>>>>> ddab61f504ab9ca072ba0f8fb38ecb8c79022acc
        reward_flag = 'gaussian_with_insulin'

        # Initialization flag: 'random' or 'fixed'
        bg_init_flag = 'random'
        max_insulin_action = 300

        return meal_times, meal_amounts, reward_flag, bg_init_flag, max_insulin_action
