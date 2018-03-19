import random

import torch
from torch import nn
from torch.autograd import variable

from .agent import Agent


class DQNAgent(Agent):
    def __init__(self, network, eps_decay):
        self.network = network
        self.epsilon = eps_decay
        self.rewards = []
        self.actions = []
        self.states = []

    def act(self, state):
        action = self.network(state)
        self.states.append(state)
        # TODO: all

    def observe(self, reward):
        # TODO: implement
        pass

    def learn(self):
        # TODO: implement
        pass

    def generator_data(self):
        # TODO: implement
        pass
