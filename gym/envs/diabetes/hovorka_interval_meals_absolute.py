"""
OPENAI gym environment for the Hovorka model
Actions runs for a longer interval (default 30 mins)
to get closer to a markov decision process.

This model includes a meal
"""

import logging
import gym
from gym import spaces

import numpy as np
# import numpy.matlib

# Plotting for the rendering
# import matplotlib.pyplot as plt

# Hovorka simulator
from gym.envs.diabetes.hovorka_model import hovorka_parameters, hovorka_model, hovorka_model_tuple
from gym.envs.diabetes.reward_function import calculate_reward

# ODE solver stuff
from scipy.integrate import ode
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)

class HovorkaMealsAbsolute(gym.Env):
    # TODO: fix metadata??
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        """
        Initializing the simulation environment.
        """

        # Action space (increase .1, decrease .1, or do nothing)
        # self.action_space = spaces.Discrete(3)

        # Continuous action space
        self.action_space = spaces.Box(0, 15, 1)
        self.previous_action = 0

        # Observation space -- bg between 0 and 500, measured every five minutes (1440 mins per day / 5 = 288)
        # self.observation_space = spaces.Box(0, 500, 288)

        self.observation_space = spaces.Box(0, 500, 60)
        # self.observation_space = spaces.Box(0, 500, 1)

        # Initial glucose regulation parameters
        self.bolus = 0

        # Initial basal -- this rate dictates the initial BG value
        self.init_basal = np.random.choice(np.linspace(4, 6.428, 50))
        # self.init_basal = np.random.choice(np.concatenate((np.linspace(.5, 4, 15), np.arange(4, 7, .5), np.arange(7,15, 1))), 1)
        # self.init_basal = 6

        # Flag for resetting
        self.reset_basal_manually = None

        self._seed()
        self.viewer = None

        # ==========================================
        # Setting up the Hovorka simulator
        # ==========================================

        # Patient parameters
        P = hovorka_parameters(70)
        self.P = P

        # Initial values for parameters
        initial_pars = (self.init_basal, 0, P)

        # Initial value
        X0 = fsolve(hovorka_model_tuple, np.zeros(10), args=initial_pars)
        self.X0 = X0

        # Simulation setup
        self.integrator = ode(hovorka_model)
        self.integrator.set_integrator('vode', method='bdf', order=5)
        self.integrator.set_initial_value(X0, 0)

        # Simulation time in minutes
        self.simulation_time = 30

        # State is BG, simulation_state is parameters of hovorka model
        # self.state = [X0[4] * 18 / P[12], X0[6]]
        # self.state = [X0[4] * 18 / P[12]]
        initial_bg = X0[4] * 18 / P[12]
        initial_insulin = X0[6]
        self.state = np.concatenate([np.repeat(initial_bg, self.simulation_time), np.repeat(initial_insulin, self.simulation_time)])

        self.simulation_state = X0

        # Keeping track of entire blood glucose level for each episode
        # self.bg_history = [X0[4] * 18 / P[12]]
        # self.insulin_history = [X0[6]]
        self.bg_history = []
        self.insulin_history = []

        # ====================
        # Meal setup
        # ====================
        meal_times = [600]
        meal_amounts = [50]

        eating_time = 30
        premeal_bolus_time = 30

        # Meals indicates the number of carbs taken at time t
        meals = np.zeros(2880)
        # 'meal_indicator' indicates time of bolus
        meal_indicator = np.zeros(2880)

        for i in range(len(meal_times)):
            meals[meal_times[i] : meal_times[i] + eating_time] = meal_amounts[i]/eating_time * 1000 /180
            meal_indicator[meal_times[i]-premeal_bolus_time:meal_times[i]] = meal_amounts[i] * 1000 / 180

        self.meals = meals
        self.meal_indicator = meal_indicator
        self.eating_time = eating_time

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        self.max_iter = 3000

        # Reward flag
        self.reward_flag = 'absolute'

        self.steps_beyond_done = None


    def _step(self, action):
        """
        Take action. In the diabetes simulation this means increase, decrease or do nothing
        to the insulin to carb ratio (bolus).
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Check for CHO and adding bolus
        # self.integrator.integrate(self.integrator.t + 1)

        self.integrator.set_initial_value(self.simulation_state, self.num_iters)

        bg = []
        insulin = []
        # ==========================
        # Integration loop
        # ==========================
        for i in range(self.simulation_time):

            # ===============================================
            # Solving one step of the Hovorka model
            # ===============================================

            insulin_rate = action + (self.meal_indicator[self.num_iters] * self.bolus)/self.eating_time
            self.integrator.set_f_params(insulin_rate, self.meals[self.num_iters], self.P)

            self.integrator.integrate(self.integrator.t + 1)

            self.num_iters += 1
            bg.append(self.integrator.y[4] * 18 / self.P[12])
            insulin.append(self.integrator.y[6])

        # Updating environment parameters
        self.simulation_state = self.integrator.y

        # Recording bg history for plotting
        self.bg_history = np.concatenate([self.bg_history, bg])
        self.insulin_history.append(self.integrator.y[6])

        # Updating state
        # self.state[0] = bg
        # self.state[1] = self.integrator.y[6]

        self.state = np.concatenate([bg, insulin])

        #Set environment done = True if blood_glucose_level is negative
        done = 0

        if (np.max(bg) > self.bg_threshold_high or np.max(bg) < self.bg_threshold_low):
            done = 1

        if self.num_iters > self.max_iter:
            done = 1

        done = bool(done)

        # ====================================================================================
        # Calculate Reward  (and give error if action is taken after terminal state)
        # ====================================================================================

        if not done:
            reward = calculate_reward(np.array(bg), self.reward_flag, 108)

        elif self.steps_beyond_done is None:
            # Blood glucose below zero -- simulation out of bounds
            self.steps_beyond_done = 0
            # reward = 0.0
            reward = -1000
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000

        self.previous_action = action

        return np.array(self.state), np.mean(reward), done, {}


    def _reset(self):
        #TODO: Insert init code here!

        # re init -- in case the init basal has been changed
        if self.reset_basal_manually is None:
            # self.init_basal = np.random.choice(np.concatenate((np.linspace(.5, 4, 15), np.arange(4, 7, .5), np.arange(7,15, 1))), 1)
            # slf.init_basal = np.random.normal(5, .6)
            self.init_basal = np.random.choice(np.linspace(4, 6.428, 50))
        else:
            self.init_basal = self.reset_basal_manually

        P = self.P
        initial_pars = (self.init_basal, 0, P)

        X0 = fsolve(hovorka_model_tuple, np.zeros(10), args=initial_pars)
        self.X0 = X0
        self.integrator.set_initial_value(self.X0, 0)

        # State is BG, simulation_state is parameters of hovorka model
        # self.state[0] = X0[4] * 18 / P[12]
        # self.state[1] = X0[6]
        initial_bg = X0[4] * 18 / P[12]
        initial_insulin = X0[6]
        self.state = np.concatenate([np.repeat(initial_bg, self.simulation_time), np.repeat(initial_insulin, self.simulation_time)])

        self.simulation_state = X0
        # self.bg_history = [X0[4] * 18 / P[12] ]
        # self.insulin_history = [X0[6]]
        self.bg_history = []
        self.insulin_history = []

        self.num_iters = 0
        # self.init_basal = np.random.choice(range(1, 10), 1)


        # changing observation space if simulation time is changed
        if self.simulation_time != 30:
            self.observation_space = spaces.Box(0, 500, self.simulation_time*2)


        self.steps_beyond_done = None
        return np.array(self.state)


    def _render(self, mode='human', close=False):
        #TODO: Clean up plotting routine

        return None
        # if mode == 'rgb_array':
            # return None
        # elif mode is 'human':
            # if not bool(plt.get_fignums()):
                # plt.ion()
                # self.fig = plt.figure()
                # self.ax = self.fig.add_subplot(111)
                # # self.line1, = ax.plot(self.bg_history)
                # self.ax.plot(self.bg_history)
                # plt.show()
            # else:
                # # self.line1.set_ydata(self.bg_history)
                # # self.fig.canvas.draw()
                # self.ax.clear()
                # self.ax.plot(self.bg_history)

            # plt.pause(0.0000001)
            # plt.show()

            # # return None
        # else:
            # super(HovorkaInterval, self).render(mode=mode) # just raise an exception

            # plt.ion()
            # plt.plot(self.bg_history)

