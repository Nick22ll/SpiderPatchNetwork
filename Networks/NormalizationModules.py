import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNorm(nn.Module):
    """
        Param: []
    """

    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, g, x):
        graph_size = g.batch_num_nodes().tolist() if self.is_node else g.batch_num_edges().tolist()
        x_list = torch.split(x, graph_size)
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


# Adjance norm for node
class AdjaNodeNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdjaNodeNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def message_func(self, edges):
        return {"h": edges.src["norm_h"]}

    def reduce_func(self, nodes):
        dst_h = nodes.mailbox['h']
        src_h = nodes.data['norm_h']

        h = torch.cat([dst_h, src_h.unsqueeze(1)], 1)
        mean = torch.mean(h, dim=(1, 2))
        var = torch.std(h, dim=(1, 2))

        mean = mean.unsqueeze(1).expand_as(src_h)
        var = var.unsqueeze(1).expand_as(src_h)
        return {"norm_mean": mean, "norm_var": var}

    def forward(self, g, h):
        g.ndata["norm_h"] = h
        g.update_all(self.message_func, self.reduce_func)

        mean = g.ndata['norm_mean']
        var = g.ndata['norm_var']

        norm_h = (h - mean) / (var + self.eps)

        if self.affine:
            return self.gamma * norm_h + self.beta
        else:
            return norm_h


# Adjance norm for edge
class AdjaEdgeNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdjaEdgeNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def message_func(self, edges):
        return {"e": edges.data['norm_e']}

    def reduce_func(self, nodes):
        e = nodes.mailbox['e']
        mean = torch.mean(e, dim=(1, 2))
        var = torch.std(e, dim=(1, 2))

        mean = mean.unsqueeze(1).expand_as(e[:, 0, :])
        var = var.unsqueeze(1).expand_as(e[:, 0, :])
        return {"norm_mean": mean, "norm_var": var}

    def apply_edges(self, edges):
        mean = edges.dst['norm_mean']
        var = edges.dst['norm_var']
        return {"edge_mean": mean, "edge_var": var}

    def forward(self, g, e):
        g.edata['norm_e'] = e
        g.update_all(self.message_func, self.reduce_func)
        g.apply_edges(self.apply_edges)
        mean = g.edata['edge_mean']
        var = g.edata['edge_var']

        norm_e = (e - mean) / (var + self.eps)
        if self.affine:
            return self.gamma * norm_e + self.beta
        else:
            return norm_e


class UnitedNormBase(nn.Module):

    def __init__(self, num_features, is_node=True):
        super(UnitedNormBase, self).__init__()
        self.clamp = False
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(self.num_features))
        self.beta = nn.Parameter(torch.zeros(self.num_features))

        self.lambda_batch = nn.Parameter(torch.ones(self.num_features))
        self.lambda_graph = nn.Parameter(torch.ones(self.num_features))
        self.lambda_adja = nn.Parameter(torch.ones(self.num_features))
        self.lambda_node = nn.Parameter(torch.ones(self.num_features))

        self.lambdas = [self.lambda_batch, self.lambda_graph, self.lambda_adja, self.lambda_node]
        self.norm_names = ["BATCH", "GRAPH", "ADJACENT", "NODE"]

        self.batch_norm = nn.BatchNorm1d(self.num_features, affine=False)
        self.graph_norm = GraphNorm(self.num_features, is_node=is_node, affine=False)
        self.node_norm = nn.LayerNorm(self.num_features, elementwise_affine=False)
        if is_node:
            self.adja_norm = AdjaNodeNorm(self.num_features, affine=False)
        else:
            self.adja_norm = AdjaEdgeNorm(self.num_features, affine=False)

    def norm_lambda(self):
        raise NotImplementedError

    def forward(self, g, x):
        x_b = self.batch_norm(x)
        x_g = self.graph_norm(g, x)
        x_a = self.adja_norm(g, x)
        x_n = self.node_norm(x)

        lambda_batch, lambda_graph, lambda_adja, lambda_node = self.norm_lambda()
        x_new = lambda_batch * x_b + lambda_graph * x_g + lambda_adja * x_a + lambda_node * x_n
        return self.gamma * x_new + self.beta


class UnitedNormCommon(UnitedNormBase):

    def __init__(self, *args):
        super(UnitedNormCommon, self).__init__(*args)
        self.clamp = True

    def norm_lambda(self):
        lambda_sum = self.lambda_batch + self.lambda_graph + self.lambda_adja + self.lambda_node
        return self.lambda_batch / lambda_sum, \
               self.lambda_graph / lambda_sum, \
               self.lambda_adja / lambda_sum, \
               self.lambda_node / lambda_sum


class UnitedNormSoftmax(UnitedNormBase):

    def __init__(self, *args):
        super(UnitedNormSoftmax, self).__init__(*args)
        self.clamp = False

    def norm_lambda(self):
        concat_lambda = torch.cat([self.lambda_batch.unsqueeze(0), \
                                   self.lambda_graph.unsqueeze(0), \
                                   self.lambda_adja.unsqueeze(0), \
                                   self.lambda_node.unsqueeze(0)], dim=0)

        softmax_lambda = F.softmax(concat_lambda, dim=0)
        return softmax_lambda[0], softmax_lambda[1], softmax_lambda[2], softmax_lambda[3]
