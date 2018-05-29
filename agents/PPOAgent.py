import torch
from torch import optim
from torch.nn import functional as F
import copy

from .agent import Agent
from utils import device, gumbel_noise_like


class PPOAgent(Agent):
    def __init__(self, num_actions, network, lr=1e-4, discount=0.99, beta=0.01, eps=0.1, asynchronous=False):
        self.num_actions = num_actions
        self.network = network
        self.old_network = copy.deepcopy(self.network)
        self.optimizer = None
        self.discount = discount
        self.beta = beta
        self.eps = eps
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.old_policies = []

        if not asynchronous:
            self.optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0001)

        self.old_network.flatten_parameters()

    def async_optim(self, optimizer):
        self.optimizer = optimizer

    def reset(self):
        self.network.reset_state()
        self.old_network.reset_state()
        self.old_network.load_state_dict(self.network.state_dict())

        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.old_policies = []

    def act(self, state, deterministic=False):
        # stop gradient to state
        policy, value = self.network(state.detach())
        old_policy, _ = self.old_network(state.detach())

        sq_policy = old_policy.squeeze()
        if not deterministic:
            gumbel = gumbel_noise_like(sq_policy)
            action = torch.argmax(sq_policy + gumbel, dim=-1)
        else:
            action = torch.argmax(sq_policy, dim=-1)

        self.actions.append(action)
        self.policies.append(F.softmax(policy, dim=-1))
        self.values.append(value)
        self.old_policies.append(F.softmax(old_policy, dim=-1))

        return action

    def observe(self, reward):
        self.rewards.append(reward)

    def learn(self, device=device):
        num_steps = len(self.rewards)
        # discount reward over whole episode
        r = 0.
        rewards = torch.zeros((num_steps, self.rewards[0].size(0)), device=device)
        for n in reversed(range(num_steps)):
            rewards[n, :] = r = self.rewards[n] + self.discount * r

        rewards = rewards.view(-1)
        actions = torch.cat(self.actions).view(-1)
        policy = torch.cat(self.policies).view(-1, self.num_actions)
        value = torch.cat(self.values).view(-1)
        old_policy = torch.cat(self.old_policies). view(-1, self.num_actions).detach()

        indices = torch.arange(policy.size(0), dtype=torch.long)

        advantage = rewards - value
        ratio = policy[indices, actions] / (old_policy[indices, actions] + 1e-8)
        # MSE on rewards and values
        loss = 0.5 * torch.mean(torch.pow(advantage, 2.))
        # clipped surrogate objective
        loss += torch.mean(torch.min(-ratio * advantage.detach(),
                                     -torch.clamp(ratio, 1. - self.eps, 1. + self.eps) * advantage.detach()))
        # entropy pentalty
        # loss += self.beta * torch.mean(torch.sum(policy * torch.log(policy + 1e-8), dim=-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()

        return loss.item(), torch.mean(value).item()
