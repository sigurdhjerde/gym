"""
OPENAI gym environment for the incremental Hovorka model
Converted/inspired from cartpole to Hovorka model!

The actions are: increase insulin by 20%, decrease by 20%, do nothing, reset to init basal
and set insulin to zero.
"""

import logging
import gym
from gym import spaces
# from gym.utils import seeding
import numpy as np
import numpy.matlib

# Plotting for the rendering
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Hovorka simulator
from gym.envs.diabetes import hovorka_simulator as hs
# import hovorka_simulator as hs

# ODE solver stuff
# from scipy.integrate import ode
# from scipy.optimize import fsolve

logger = logging.getLogger(__name__)

class HovorkaDiabetesIncremental(gym.Env):
    # TODO: fix metadata
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
        # self.action_space = spaces.Box(0, 200, 1)
        self.action_space = spaces.Discrete(5)

        # Observation space -- bg between 0 and 500, measured every five minutes (1440 mins per day / 5 = 288)
        # self.observation_space = spaces.Box(0, 500, 288)

        self.observation_space = spaces.Box(0, 500, 2)

        # Initial glucose regulation parameters
        self.basal = 8.3
        self.bolus = 8.8
        self.init_basal = 6.66
        # self.init_basal = np.random.choice(np.concatenate((np.linspace(.5, 4, 15), np.arange(4, 7, .5), np.arange(7,15, 1))), 1)

        self._seed()
        self.viewer = None

        # Initial state
        X0, _, _, _, P = hs.simulation_setup(1, self.init_basal)

        # State is BG and plasma insulin concentration, simulation_state is parameters of hovorka model
        self.state = [X0[4] * 18 / P[12], X0[6]]
        self.simulation_state = X0

        # self.state[0] = X0[4] * 18 / P[12]
        # self.state[1] = X0[6]

        # Keeping track of entire blood glucose level for each episode
        self.bg_history = [X0[4] * 18 / P[12]]
        self.insulin_history = [X0[6]]

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        # Arbitrary number to end episode
        self.max_iter = 3000

        self.steps_beyond_done = None

    def _step(self, action):
        """
        Take action. In the diabetes simulation this means increase, decrease or do nothing
        to the insulin to carb ratio (bolus).
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Variables
        simulation_state = self.simulation_state
        IC = self.bolus

        if action == 0:
            self.basal = self.basal * .8
        elif action == 1:
            self.basal = self.basal * 1.2
        elif action == 2:
            self.basal = self.basal * 1
        elif action == 3:
            self.basal = 0
        elif action == 4:
            self.basal = 8.3


        # Hovorka model with meals
        bg, simulation_state_new = hs.simulate_one_step_with_meals(self.basal, IC, self.num_iters, simulation_state)
        self.num_iters += 1

        # Updating environment parameters
        self.simulation_state = simulation_state_new
        # self.bolus = bolus_new
        # self.basal = action
        self.bg_history.append(bg)
        self.insulin_history.append(simulation_state_new[6])

        # Updating state
        self.state[0] = bg
        self.state[1] = simulation_state_new[6]

        #Set environment done = True if blood_glucose_level is negative
        done = 0

        if (bg > self.bg_threshold_high or bg < self.bg_threshold_low):
            done = 1

        if self.num_iters > self.max_iter:
            done = 1

        done = bool(done)

        # Calculate Reward  (and give error if action is taken after terminal state)
        if not done:
            reward = hs.calculate_reward(self.state[0])
        elif self.steps_beyond_done is None:
            # Blood glucose below zero -- simulation out of bounds
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        #TODO: Insert init code here!

        # Initial state using cont measurements
        X0, _, _, _, P = hs.simulation_setup(1, self.init_basal)

        # State is BG, simulation_state is parameters of hovorka model
        self.state[0] = X0[4] * 18 / P[12]
        self.state[1] = X0[6]
        self.simulation_state = X0

        self.bg_history = [X0[4] * 18 / P[12] ]
        self.insulin_history = [X0[6]]

        self.num_iters = 0
        self.basal = 8.3
        # self.init_basal = np.random.choice(range(1, 10), 1)
        # self.init_basal = np.random.choice(np.concatenate((np.linspace(.5, 4, 15), np.arange(4, 7, .5), np.arange(7,15, 1))), 1)
        # self.basal = self.init_basal


        self.steps_beyond_done = None
        return np.array(self.state)


    def _render(self, mode='human', close=False):
        #TODO: Clean up plotting routine

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
            super(HovorkaDiabetesIncremental, self).render(mode=mode) # just raise an exception

            plt.ion()
            plt.plot(self.bg_history)
