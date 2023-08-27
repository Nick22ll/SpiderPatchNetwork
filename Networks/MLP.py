import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(MLP, self).__init__()

        ####  Layers  ####
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

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
        self.dropout = dropout

    def forward(self, x):
        # FIRST LAYER
        x = self.LeakyReLU(self.norm1(self.in_linear(x)))

        x = torch.nn.functional.dropout(x, torch.rand(1).item() * self.dropout, training=self.training)

        # SECOND LAYER
        x = self.LeakyReLU(self.norm2(self.hidden_linear(x)))

        x = torch.nn.functional.dropout(x, torch.rand(1).item() * self.dropout, training=self.training)

        # CLASSIFIER
        return self.classifier(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

