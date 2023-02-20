import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()

        ####  Layers  ####
        self.in_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

        ####  Normalization Layers  ####
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.in_linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.hidden_linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # FIRST LAYER
        x = self.in_linear(x)
        x = self.norm1(x)
        x = self.LeakyReLU(x)

        x = self.dropout(x)

        # SECOND LAYER
        x = self.hidden_linear(x)
        x = self.norm2(x)
        x = self.LeakyReLU(x)

        x = self.dropout(x)

        # CLASSIFIER
        return self.classifier(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

