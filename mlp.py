from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, d_input, d_hidden, d_output):

        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)
        self.linear3 = nn.Linear(d_hidden, d_output)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
