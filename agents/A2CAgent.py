import copy
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable

from .agent import Agent
from utils import cudify, one_hot, gumbel_noise


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

        sq_policy = policy.squeeze()
        gumbel = Variable(gumbel_noise(sq_policy.shape))
        _, action = torch.max(sq_policy + gumbel, dim=-1)

        self.states.append(state)
        self.actions.append(action)
        self.policies.append(F.softmax(policy, dim=-1))
        self.values.append(value)

        self.g_policies.append(F.softmax(g_policy, dim=-1))
        self.g_values.append(g_value)

        return action.data.cpu().numpy()

    def observe(self, reward):
        self.rewards.append(reward)

    def learn(self):
        num_steps = len(self.rewards)
        # discount reward over whole episode
        r = 0.
        rewards = np.zeros((num_steps, self.states[0].shape[0]))
        for n in reversed(range(num_steps)):
            rewards[n, :] = r = self.rewards[n] + self.discount * r

        rewards = Variable(cudify(torch.from_numpy(np.array(rewards.flatten(), dtype=np.float32))))
        actions = one_hot(torch.cat(self.actions), num_classes=self.num_actions)
        policy = torch.cat(self.policies).view(-1, self.num_actions)
        value = torch.cat(self.values).view(-1)

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

        policy = torch.cat(self.g_policies)
        value = torch.cat(self.g_values)
        rewards = Variable(cudify(torch.from_numpy(np.array(self.rewards))))

        # entropy = torch.mean(torch.sum(-policy * torch.log(policy + 1e-8), dim=1))

        self.reset_generator()
        return policy, value, rewards
