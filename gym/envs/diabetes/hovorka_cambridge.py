"""
OPENAI gym environment for the Hovorka model using the cambridge parameter set

The reason we are doing this is that the cambridge model is very slow!

This is the base class for the Hovorka models.
    - Actions runs for a longer interval (default 30 mins)
    to get closer to a markov decision process.
    - The model includes meals
    - Default 34 dim observation space (30 min BG and last four actions)
    - Default action space 0 to 50 mU/min of insulin
    - Rendering disabled by default

    - Initialization and reset: Random initialization and no meals!


    - TODO:
        - Double check parameters - renal threshold et al
        - Check difference in previous insulin compared to cambridge model
        - check initial basal rates

"""

import logging
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.diabetes.meal_generator.meal_generator import meal_generator

import numpy as np

# Plotting for the rendering
import matplotlib.pyplot as plt

# Hovorka simulator
from gym.envs.diabetes.hovorka_model import hovorka_model, hovorka_model_tuple
# from gym.envs.diabetes.reward_function import calculate_reward
from gym.envs.diabetes.hovorka_cambride_pars import hovorka_cambridge_pars
from gym.envs.diabetes.reward_function import RewardFunction

# ODE solver stuff
from scipy.integrate import ode
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)

rewardFunction = RewardFunction()


class HovorkaCambridgeBase(gym.Env):
    # TODO: fix metadata??
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        """
        Initializing the simulation environment.
        """

        self.previous_action = 0

        # State space
        # self.observation_space = spaces.Box(0, 500, (34,), dtype=np.float32)
        # self.observation_space = spaces.Box(0, 500, 1)

        # Bolus rate -- [mU/mmol]
        # self.bolus = 0
        # self.bolus = 8.3

        # Bolus carb factor -- [g/U]
        self.bolus = 25
        self.meal_seed = None

        ## Loading variable parameters
        # meal_times, meal_amounts, reward_flag, bg_init_flag, max_insulin_action = self._update_parameters()
        reward_flag, bg_init_flag = self._update_parameters()


        # Action space
        self.action_space = spaces.Box(0, 50, (1,), dtype=np.float32)

        # Initialize bolus history
        self.bolusHistoryIndex = 0
        self.bolusHistoryValue = []
        self.bolusHistoryTime = []

        # Initialize sensor model
        self.CGMlambda = 15.96    # Johnson parameter of recalibrated and synchronized sensor error.
        self.CGMepsilon = -5.471  # Johnson parameter of recalibrated and synchronized sensor error.
        self.CGMdelta = 1.6898    # Johnson parameter of recalibrated and synchronized sensor error.
        self.CGMgamma = -0.5444   # Johnson parameter of recalibrated and synchronized sensor error.
        self.CGMerror = 0
        # self.sensorNoiseValue = 0 # Set a value

        # ====================================
        # Normalized action space!!
        # ====================================
        # self.action_space = spaces.Box(-1, 1, (1,))

        # Increasing the max bolus rate
        # self.action_space = spaces.Box(0, 150, 1)

        # Cambridge parameters
        P, init_basal_optimal = hovorka_cambridge_pars(0)
        self.P = P
        self.init_basal_optimal = init_basal_optimal

        # Initial basal -- this rate dictates the initial BG value

        if bg_init_flag == 'random':
            # self.init_basal = np.random.choice(np.linspace(4, 6.428, 50))
            self.init_basal = np.random.choice(np.linspace(init_basal_optimal-2, init_basal_optimal, 10))
        elif bg_init_flag == 'fixed':
            self.init_basal = init_basal_optimal

        # Flag for manually resetting the init
        self.reset_basal_manually = None

        self._seed()
        self.viewer = None

        # ==========================================
        # Setting up the Hovorka simulator
        # ==========================================

        # Initial values for parameters
        initial_pars = (self.init_basal, 0, P)

        # Initial value
        X0 = fsolve(hovorka_model_tuple, np.zeros(11), args=initial_pars)
        self.X0 = X0

        # Simulation setup
        self.integrator = ode(hovorka_model)
        self.integrator.set_integrator('vode', method='bdf', order=5)
        # self.integrator.set_integrator('lsoda', method='bdf')
        # self.integrator.set_integrator('dopri853')

        self.integrator.set_initial_value(X0, 0)

        # Simulation time in minutes
        self.simulation_time = 30
        self.n_solver_steps = 1
        self.stepsize = int(self.simulation_time/self.n_solver_steps)
        # self.observation_space = spaces.Box(0, 500, (int(self.stepsize + 4 + 1),), dtype=np.float32)
        self.observation_space = spaces.Box(0, 500, (int(self.stepsize + 4 ),), dtype=np.float32)

        # State is BG, simulation_state is parameters of hovorka model
        initial_bg = X0[-1] * 18
        initial_insulin = np.zeros(4)
        initial_iob = np.zeros(1)
        # self.state = np.concatenate([np.repeat(initial_bg, self.simulation_time), initial_insulin, initial_iob])
        self.state = np.concatenate([np.repeat(initial_bg, self.simulation_time), initial_insulin])

        self.simulation_state = X0

        # Keeping track of entire blood glucose level for each episode
        self.bg_history = []
        self.insulin_history = initial_insulin
        # self.insulin_history = []

        # ====================
        # Meal setup
        # ====================

        # eating_time = self.n_solver_steps
        eating_time = 1
        meals, meal_indicator = meal_generator(eating_time=eating_time, premeal_bolus_time=0,seed=self.meal_seed)
        # meals = np.zeros(1440)
        # meal_indicator = np.zeros(1440)

        # TODO: Clean up these
        self.meals = meals
        self.meal_indicator = meal_indicator
        self.eating_time = eating_time

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        # TODO: This number is arbitrary
        self.max_iter = 1440

        # Reward flag
        self.reward_flag = reward_flag

        self.steps_beyond_done = None


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _update_parameters(self):
        ''' Update parameters of model,
        this is only used for inherited classes'''

        # meal_times = [0]
        # meal_amounts = [0]
        # reward_flag = 'gaussian'
        reward_flag = 'asymmetric'
        # reward_flag = 'binary_tight'
        bg_init_flag = 'random'
        # action_space = spaces.box(0, 30, 1)

        # return meal_times, meal_amounts, reward_flag, bg_init_flag
        return reward_flag, bg_init_flag

    def scalableExpIOB(self, t, tp, td):
            #SCALABLEEXPIOB
            # Calculates the insulin bolus on board using a decay
            # expenontiel. Function taken from
            # https://github.com/ps2/LoopIOB/blob/master/ScalableExp.ipynb
            # Original contributor Dragan Maksimovic (@dm61)
            #
            # Inputs:
            #    - t: Time duration after bolus delivery.
            #    - tp: Time of peak action of insulin.
            #    - td: Time duration of insulin action.
            #
            # For more info on tp and td:
            # http://guidelines.diabetes.ca/cdacpg_resources/Ch12_Table1_Types_of_Insulin_updated_Aug_5.pdf
            #

            if t > td:
                iob = 0
                return iob
            else:
                tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
                a = 2 * tau / td
                S = 1 / (1 - a + (1 + a) * np.exp(-td/tau))
                iob = 1 - S * (1 - a) * ((t**2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t/tau) + 1)
                return iob

    def step(self, action):
        """
        Take action. In the diabetes simulation this means increase, decrease or do nothing
        to the insulin to carb ratio (bolus).
        """
        if action > self.action_space.high:
            action = self.action_space.high
        elif action < self.action_space.low:
            action = self.action_space.low

        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Converting scaled action
        # ub = 50
        # lb = 0
        # action = lb + (action + 1) * .5 * (ub - lb)

        self.integrator.set_initial_value(self.simulation_state, self.num_iters)

        bg = []
        # insulin = []
        # ==========================
        # Integration loop
        # ==========================
        for i in range(self.simulation_time):
        # for i in range(6):

            # ===============================================
            # Solving one step of the Hovorka model
            # ===============================================

            # Add bolus to history
            if self.meal_indicator[self.num_iters] > 0:
                self.bolusHistoryIndex = self.bolusHistoryIndex + 1
                self.bolusHistoryValue.append(self.meal_indicator[self.num_iters] * (180/self.bolus))
                self.bolusHistoryTime.append(self.num_iters)
                # self.lastBolusTime = self.num_iters

            # Basal rate = action, bolus calculated from carb ratio
            # insulin_rate = action + (self.meal_indicator[self.num_iters] * (180/self.bolus) )/self.eating_time
            insulin_rate = action + (self.meal_indicator[self.num_iters] * (180/self.bolus) )

            # Setting the carb and insulin parameter in the model
            self.integrator.set_f_params(insulin_rate, self.meals[self.num_iters], self.P)

            # if self.meal_indicator[self.num_iters] > 0:
            #     # insulin_rate = action + (self.meal_indicator[self.num_iters] * (180/self.bolus) )/self.eating_time
            #     # self.integrator.set_f_params(insulin_rate, self.meals[self.num_iters], self.P)
            #     print(self.integrator.y[0])

            # Debugging TODO remove!
            # if self.meals[self.num_iters] > 0:
                # print(self.meals[self.num_iters])

            # if insulin_rate > 6.43:
                # print(insulin_rate)

            self.integrator.integrate(self.integrator.t + 1)
            # self.integrator.integrate(self.integrator.t + 5)
            # print(self.integrator.y[0])


            # Only updating the cgm every 'n_solver_steps' minute
            # if np.mod(i, self.n_solver_steps)==0:
                # bg.append(self.integrator.y[-1] * 18)

            # ===============
            # CGM noise
            # ===============

            # johnson
            sensor_noise = 0.7 * (self.CGMerror + np.random.randn(1))
            # paramMCHO = 180
            self.CGMerror = (10 / 180) * (self.CGMepsilon + self.CGMlambda *
                                          np.sinh((sensor_noise[0] - self.CGMgamma) / self.CGMdelta))

            # # ar(1), colored}
            # phi = 0.8
            # self.CGMerror = phi * self.CGMerror + np.sqrt(1 - phi ^ 2) * self.sensorNoiseValue * np.randn(1)

            # # mult
            # self.CGMerror = self.sensorNoiseValue * self.state(self.integrator.y[-1]) * np.random.randn(1)

            # # white, add
            # self.CGMerror = self.sensorNoiseValue * np.random.randn(1)

            # # No noise
            # self.CGMerror = 0

            # bg.append(self.integrator.y[-1] * 18)
            bg.append(self.integrator.y[-1] * 18 + self.CGMerror)

            # self.num_iters += 5
            self.num_iters += 1

        # Updating environment parameters
        self.simulation_state = self.integrator.y

        # Recording bg history for plotting
        self.bg_history = np.concatenate([self.bg_history, bg])
        self.insulin_history = np.concatenate([self.insulin_history, insulin_rate])

        # Updating state

        insulinOnBoard = np.zeros(1)
        if self.bolusHistoryIndex > 0:
           for b in range(self.bolusHistoryIndex):
               insulinOnBoard = insulinOnBoard + self.bolusHistoryValue[b] * self.scalableExpIOB(self.num_iters - self.bolusHistoryTime[b], 75, 240)

        # self.state = np.concatenate([bg, list(reversed(self.insulin_history[-4:])), insulinOnBoard])
        self.state = np.concatenate([bg, list(reversed(self.insulin_history[-4:]))])

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
            if self.reward_flag != 'gaussian_with_insulin':
                reward = rewardFunction.calculate_reward(np.array(bg), self.reward_flag, 108)
            else:
                reward = rewardFunction.calculate_reward(np.array(bg), 'gaussian_with_insulin', 108, action)

        elif self.steps_beyond_done is None:
            # Blood glucose below zero -- simulation out of bounds
            self.steps_beyond_done = 0
            # reward = 0.0
            # reward = -1000
            if self.reward_flag != 'gaussian_with_insulin':
                reward = rewardFunction.calculate_reward(np.array(bg), self.reward_flag, 108)
            else:
                reward = rewardFunction.calculate_reward(np.array(bg), 'gaussian_with_insulin', 108, action)
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000

        self.previous_action = action

        return np.array(self.state), np.mean(reward), done, {}


    def reset(self):
        #TODO: Insert init code here!

        # re init -- in case the init basal has been changed
        if self.reset_basal_manually is None:
            # self.init_basal = np.random.choice(np.linspace(4, 6.428, 50))
            self.init_basal = np.random.choice(np.linspace(self.init_basal_optimal-2, self.init_basal_optimal, 10))
            # self.init_basal = 6
        else:
            self.init_basal = self.reset_basal_manually

        P = self.P
        initial_pars = (self.init_basal, 0, P)

        X0 = fsolve(hovorka_model_tuple, np.zeros(11), args=initial_pars)
        self.X0 = X0
        self.integrator.set_initial_value(self.X0, 0)

        # State is BG, simulation_state is parameters of hovorka model
        initial_bg = X0[-1] * 18
        initial_insulin = np.zeros(4)
        initial_iob = np.zeros(1)
        # self.state = np.concatenate([np.repeat(initial_bg, self.simulation_time/self.n_solver_steps), initial_insulin, initial_iob])
        self.state = np.concatenate([np.repeat(initial_bg, self.stepsize), initial_insulin])

        self.simulation_state = X0
        self.bg_history = []
        self.insulin_history = initial_insulin
        # self.insulin_history = []

        self.num_iters = 0


        # changing observation space if simulation time is changed
        # if self.simulation_time != 30:
        # if self.stepsize != 1:
            # observation_space_shape = int(self.stepsize + 4 + 1)
            # self.observation_space = spaces.Box(0, 500, (observation_space_shape,), dtype=np.float32)


        self.steps_beyond_done = None
        return np.array(self.state)


    def render(self, mode='human', close=False):
        #TODO: Clean up plotting routine

        # return None
        if mode == 'rgb_array':
            return None
        elif mode is 'human':
            if not bool(plt.get_fignums()):
                plt.ion()
                self.fig = plt.figure()
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
            super(HovorkaCambridgeBase, self).render(mode=mode) # just raise an exception

            plt.ion()
            plt.plot(self.bg_history)
