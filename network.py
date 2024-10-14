import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q Network"""
    def __init__(self, input_shape, n_actions, device, hidden_dim=128):
        super(DQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(input_shape, self.hidden_dim, device=device)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim, device=device)
        self.layer3 = nn.Linear(self.hidden_dim, n_actions, device=device)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        q = self.layer3(x)
        return q


class RNN_AGENT(nn.Module):
    """RNN Network"""
    def __init__(self, input_shape, n_actions, hidden_dim, device):
        super(RNN_AGENT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(input_shape, self.hidden_dim, device=device)
        self.layer2 = nn.GRUCell(self.hidden_dim, self.hidden_dim, device=device)
        self.layer3 = nn.Linear(self.hidden_dim, n_actions, device=device)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.layer1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, x, h_):
        x = F.relu(self.layer1(x))
        h_in = h_.reshape(-1, self.hidden_dim)
        h = self.layer2(x, h_in)
        q = self.layer3(h)
        return q, h

