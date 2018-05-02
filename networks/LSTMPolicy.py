from torch import nn
from torch.nn import functional as F


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, num_actions, gumbel=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2)
        self.policy = nn.Linear(self.lstm.hidden_size, num_actions)
        self.value = nn.Linear(self.lstm.hidden_size, 1)

        self.gumbel = gumbel
        self.state = None

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def reset_state(self):
        self.state = None

    def forward(self, inputs):
        h = inputs.view(1, -1, self.lstm.input_size)
        out, self.state = self.lstm(h, self.state)
        policy = self.policy(out)
        if not self.gumbel:
            policy = F.softmax(policy, dim=-1)
        value = self.value(out)
        return policy, value

