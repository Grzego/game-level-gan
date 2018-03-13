import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from .agent import Agent


class A2CAgent(Agent):
    def __init__(self, num_actions, network, lr=1e-4, discount=0.99, beta=0.01):
        super(A2CAgent, self).__init__()
        self.num_actions = num_actions
        self.network = network
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
        policy, value = self.network(state.detach())
        g_policy, g_value = self.network(state)

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

    def learn(self, use_cuda=True):
        num_steps = len(self.rewards)
        # discount reward over whole episode
        r = 0.
        rewards = [0] * len(self.rewards)
        for n in reversed(range(num_steps)):
            rewards[n] = r = self.rewards[n] + self.discount * r

        rewards = Variable(torch.from_numpy(np.array(rewards, dtype=np.float32)))
        rewards = rewards.cuda() if use_cuda else rewards

        actions = Variable(torch.from_numpy(np.eye(self.num_actions, dtype=np.float32)[np.array(self.actions)]))
        actions = actions.cuda() if use_cuda else actions

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
        # value = torch.cat(self.g_values).squeeze()

        total_reward = sum(self.rewards)

        entropy = torch.mean(torch.sum(-policy * torch.log(policy + 1e-8), dim=1))
        self.reset_generator()
        return total_reward * entropy
