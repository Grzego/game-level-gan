import copy
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from .agent import Agent
from utils import cudify


class A2CAgent(Agent):
    def __init__(self, num_actions, network, lr=1e-4, discount=0.99, beta=0.01):
        super(A2CAgent, self).__init__()
        self.num_actions = num_actions
        self.network = network
        self.network_copy = copy.deepcopy(self.network)
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.discount = discount
        self.beta = beta
        self.states = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.g_policies = []
        self.g_values = []

    def reset(self):
        self.network.reset_state()
        self.network_copy.reset_state()
        self.network_copy.load_state_dict(self.network.state_dict())

        self.states = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []

    def reset_generator(self):
        # values and policies graph nodes for generator
        self.g_policies = []
        self.g_values = []

    def act(self, state):
        # stop gradient to state
        policy, value = self.network(state.detach())
        # stop gradient to network (using its copy)
        g_policy, g_value = self.network_copy(state)

        cpu_policy = policy.cpu().squeeze().data.numpy()
        action = np.random.choice(np.arange(self.num_actions), p=cpu_policy)

        self.states.append(state)
        self.actions.append(action)
        self.policies.append(policy)
        self.values.append(value)

        self.g_policies.append(g_policy)
        self.g_values.append(g_value)

        return action

    def observe(self, reward):
        self.rewards.append(reward)

    def learn(self):
        num_steps = len(self.rewards)
        # discount reward over whole episode
        r = 0.
        rewards = [0] * len(self.rewards)
        for n in reversed(range(num_steps)):
            rewards[n] = r = self.rewards[n] + self.discount * r

        rewards = Variable(cudify(torch.from_numpy(np.array(rewards, dtype=np.float32))))
        actions = Variable(cudify(torch.from_numpy(np.eye(self.num_actions, dtype=np.float32)[np.array(self.actions)])))

        policy = torch.cat(self.policies).squeeze()
        value = torch.cat(self.values).squeeze()

        advantage = rewards - value
        # MSE on rewards and values
        loss = 0.5 * torch.sum(torch.pow(advantage, 2.))
        # CE on policy and actions
        loss -= torch.sum(advantage.detach() * torch.log(torch.sum(actions * policy, dim=1) + 1e-8))
        # entropy pentalty
        loss += self.beta * torch.sum(policy * torch.log(policy + 1e-8))
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.reset()

    def generator_data(self):
        # entropy of actions?
        # scaled by total reward?

        policy = torch.cat(self.g_policies).squeeze()
        value = torch.cat(self.g_values).squeeze()
        total_reward = sum(self.rewards)

        # entropy = torch.mean(torch.sum(-policy * torch.log(policy + 1e-8), dim=1))

        self.reset_generator()
        return policy, value, total_reward
