"""
OPENAI gym environment for the minimal model with meals
as found in the Yasini paper.
"""

import logging
import gym
from gym import spaces

import numpy as np
import numpy.matlib
from scipy.integrate import ode

# Plotting for the rendering
import matplotlib.pyplot as plt

# Hovorka simulator
from gym.envs.diabetes import minimal_model_yasini as mm
from gym.envs.diabetes.hovorka_model import hovorka_parameters
from gym.envs.diabetes.reward_function import calculate_reward


logger = logging.getLogger(__name__)

class YasiniMeals(gym.Env):
    # TODO: fix metadata
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        """
        Initializing the simulation environment.
        """

        # Continuous action space -- pump rate
        # The maximum rate on the Medtronic minimed pumps is
        # 583 mU/min. We round down to a max rate of 500
        self.action_space = spaces.Box(0, 500, 1)

        # Continuous observation space -- blood glucose and plasma insulin rate
        self.observation_space = spaces.Box(0, 500, 2)

        self._seed()
        self.viewer = None

        # =====================
        # Ode solvers
        # =====================

        # Models
        self.integrator_carb = ode(mm.carb_model)
        self.integrator_insulin = ode(mm.insulin_model)

        # Solver setup
        self.integrator_carb.set_integrator('dop853')
        self.integrator_insulin.set_integrator('dop853')

        # Initial values
        self.integrator_carb.set_initial_value(np.array([0, 0, 0]))

        self.init_bg = np.random.choice(range(70, 150, 1), 1)
        self.init_insulin = 0

        self.integrator_insulin.set_initial_value(np.array([self.init_bg, 0, self.init_insulin]))

        # Hovorka parameters
        self.P = hovorka_parameters(70)

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        self.bg_history = []
        self.insulin_history = []

        self.max_iter = 3000
        self.reward_flag = 'absolute'

        self.steps_beyond_done = None

        # ====================
        # Meal setup
        # ====================
        meal_times = [600]
        meal_amounts = [50]

        eating_time = 30

        # Meals indicates the number of carbs taken at time t
        meals = np.zeros(1440)
        # 'meal_indicator' indicates time of bolus

        for i in range(len(meal_times)):
            meals[meal_times[i] : meal_times[i] + eating_time] = meal_amounts[i]/eating_time * 1000 /180

        self.meals = meals
        self.eating_time = eating_time
        self.IC = 0


    def _step(self, action):
        """
        Set the insulin pump rate
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Running the simulation for one step

        # The action is the insulin pump ratio
        carbs = self.meals[self.num_iters]
        self.integrator_carb.set_f_params(carbs, self.P)
        self.integrator_carb.integrate(self.integrator_carb.t + 1)

        # Meal bolus
        bolus = self.meals[self.num_iters] * self.IC

        self.integrator_insulin.set_f_params(action + bolus, self.integrator_carb.y[2], self.P)
        self.integrator_insulin.integrate(self.integrator_insulin.t + 1)

        # Updating state
        bg = self.integrator_insulin.y[0]
        insulin = self.integrator_insulin.y[2]
        self.state = [bg, insulin]

        self.num_iters += 1

        # Updating environment parameters
        self.bg_history.append(bg)
        self.insulin_history.append(insulin)


        #Set environment done = True if blood_glucose_level is negative
        done = 0

        if (bg > self.bg_threshold_high or bg < self.bg_threshold_low):
            done = 1

        if self.num_iters > self.max_iter:
            done = 1

        done = bool(done)

        # ====================================================================================
        # Calculate Reward  (and give error if action is taken after terminal state)
        # ====================================================================================

        if not done:
            reward = calculate_reward(bg, self.reward_flag)
        elif self.steps_beyond_done is None:
            # Blood glucose below zero -- simulation out of bounds
            self.steps_beyond_done = 0
            reward = -1000
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000

        return np.array(self.state), reward, done, {}

    def _reset(self):
        #TODO: Insert init code here!


        # Initial values
        self.integrator_carb.set_initial_value(np.array([0, 0, 0]))

        self.init_bg = np.random.choice(range(70, 150, 1), 1)
        self.integrator_insulin.set_initial_value(np.array([self.init_bg, 0, self.init_insulin]))

        self.state = [self.init_bg, self.init_insulin]

        self.bg_history = []
        self.insulin_history = []

        self.num_iters = 0

        self.steps_beyond_done = None

        return np.array(self.state)


    def _render(self, mode='human', close=False):
        #TODO: Clean up plotting routine

        if mode == 'rgb_array':
            return None
        elif mode is 'human':
            # if not bool(plt.get_fignums()):
            if not 999 in plt.get_fignums():
                plt.ion()
                self.fig = plt.figure(999)
                self.ax = self.fig.add_subplot(111)
                # self.line1, = ax.plot(self.bg_history)
                self.ax.plot(self.bg_history)
                plt.show()
            else:
                # self.line1.set_ydata(self.bg_history)
                # self.fig.canvas.draw()
                self.ax.clear()
                self.ax.plot(self.bg_history)

            plt.pause(0.0000001)
            plt.show()

            return None
        else:
            super(MinimalDiabetes, self).render(mode=mode) # just raise an exception

            plt.ion()
            plt.plot(self.bg_history)

    def _close(self):
        ''' closing the rendering'''
        plt.close(1000)
