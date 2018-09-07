"""
OPENAI gym environment for the PADOVA model

runs a matlab and simulink session in the background
requires the PADOVA matlab simulator
"""

# Imports
import logging
import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

from gym.envs.diabetes.reward_function import calculate_reward

logger = logging.getLogger(__name__)

class PadovaDiabetes(gym.Env):
    # TODO: What does this mean!!
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Initializing the simulation environment
        """

        self.action_space = spaces.Box(0, 100, 1)

        self.observation_space = spaces.Box(0, 500, 1)

        # BElow: copied from other envs. TODO: dont know if this is needed
        self._seed()
        self.viewer = None

        # Starting the matlab engine
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(
            r'/home/jonas/Documents/git/padova_model/diabetes_reinforcement_learning_project_uit/padova_PA/',
            nargout=0)


        # Initializing Simulink model

        logger.info('initializing matlab simulation')
        self.eng.eval("simOut = init_sim;", nargout=0)

        self.bg_history = [self.eng.eval("simOut.yout{1}.Values.Data(end);", nargout=1)]


    def _step(self, action):
        """
        Take action and receive feedback plus reward
        """

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Take a step
        self.eng.eval("simOut = run_sim(simOut, " + str(action) + ", 70, 0);", nargout=0)

        # Read current bg level from matlab workspace
        bg = self.eng.eval("simOut.yout{1}.Values.Data(end);", nargout=1)

        self.bg_history.append(bg)

        # Updating state TODO: add insulin
        self.state = bg

        # Counter for number of iterations
        self.num_iters = 0

        # If blood glucose is less than zero, the simulator is out of bounds.
        self.bg_threshold_low = 0
        self.bg_threshold_high = 500

        self.max_iter = 3000

        # Reward flag
        self.reward_flag = 'absolute'

        self.steps_beyond_done = None

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
            # reward = 0.0
            reward = -1000
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000

        return np.array(self.state), reward, done, {}


    def _reset(self):
        """
        Resetting environment
        """

        # Shutting down simulink and clearing all variables
        self.eng.eval('close_sim', nargout=0)

        # Re-initialize
        self.eng.eval("simOut = init_sim;", nargout=0)

        bg = self.eng.eval("simOut.yout{1}.Values.Data(end);", nargout=1)

        self.bg_history = [bg]
        self.state = [bg]

        self.num_iters = 0

        self.steps_beyond_done = None

        return np.array(self.state)


    def _render():
        """
        """
        return None
