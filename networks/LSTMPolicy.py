from torch import nn
from torch.nn import functional as F


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, num_actions, gumbel=True):
        super().__init__()
        # self.lstm = nn.LSTM(input_size, 256, num_layers=2)
        self.hidden_size = 256
        self.lstm = nn.ModuleList([
            nn.LSTMCell(input_size, self.hidden_size),
            nn.LSTMCell(self.hidden_size, self.hidden_size)
        ])
        self.policy = nn.Linear(self.hidden_size, num_actions)
        self.value = nn.Linear(self.hidden_size, 1)

        self.gumbel = gumbel
        self.state = None

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def reset_state(self):
        self.state = None

    def forward(self, inputs):
        # h = inputs.view(1, -1, self.lstm.input_size)
        if self.state is None:
            self.state = [(inputs.new_zeros(inputs.size(0), self.hidden_size),
                           inputs.new_zeros(inputs.size(0), self.hidden_size))] * len(self.lstm)
        h = inputs
        for i, cell in enumerate(self.lstm):
            self.state[i] = cell(h, self.state[i])
            h = self.state[i][0]
        out = h
        policy = self.policy(out)
        if not self.gumbel:
            policy = F.softmax(policy, dim=-1)
        value = self.value(out)
        return policy, value

