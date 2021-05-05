import logging
import numpy
import torch
import math
import pandas as pd
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F

import model

import config


class GCN(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj_):
        x = torch.matmul(x, self.weight)
        output = adj_ @ x
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GcnNet(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCN(in_channels, config.HIDDEN_DIM)
        self.conv2 = GCN(config.HIDDEN_DIM, config.OUT_DIM)
        self.cached_result = None

    @staticmethod
    def preprocess_adj(adj):
        a = adj + torch.eye(adj.size(0))
        d = torch.diag(a.sum(dim=1))
        d_inv_sqrt = d.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        return d_inv_sqrt @ a @ d_inv_sqrt

    def forward(self, x_adj):
        x, adj = x_adj
        if self.cached_result is None:
            self.cached_result = self.preprocess_adj(adj)

        x = F.relu(self.conv1(x, self.cached_result))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, self.cached_result)
        return x


class CompletionWithGcn(model.ModelBase):
    def __init__(self, data_graph):
        super().__init__(data_graph)
        self.neural_network = GcnNet(data_graph.feature_dim)
        self.adj = None

    def set_adj(self, _adj):
        self.adj = _adj

    def run(self):
        return super().run((self.data_graph.data.x, self.adj))


class RefineAdj:
    def __init__(self, data_graph):
        self.data_graph = data_graph
        self.adj_matrix_leaner = CompletionWithGcn(data_graph)
        self.smb_adj = self.data_graph.get_label_smb_matrix()

    def run(self):
        val_indicator = config.VAL_INDICATOR
        total_metric = pd.DataFrame()
        adj = self.get_adj_matrix(True)
        self.adj_matrix_leaner.set_adj(adj)

        total_best_metric = pd.DataFrame([[0, 0, 0, 0, 'test', 0]],
                                         columns=['AUC', 'AP', 'POS_MAE', 'NEG_MAE', 'TYPE', 'EPOCH'])
        for i in range(config.ROUND):
            metric, best_metric = self.adj_matrix_leaner.run()
            metric['EPOCH'] = total_metric.shape[0] + numpy.array(
                range(metric.shape[0]))  # index error here, should split val and test
            total_metric = total_metric.append(metric)
            if total_best_metric[val_indicator][0] < best_metric[val_indicator][0]:
                total_best_metric = best_metric
            adj = self.get_adj_matrix(False)
            self.adj_matrix_leaner.set_adj(adj)
        return total_metric, total_best_metric

    def get_adj_matrix(self, init=True):
        if init:
            beta = config.SMB_RATIO
            adj1 = beta * self.smb_adj
        else:
            # self.adj_matrix_leaner.mpl.load_state_dict(torch.load("mygcn.pth"))
            self.adj_matrix_leaner.neural_network.eval()
            with torch.no_grad():
                z = self.adj_matrix_leaner.neural_network((self.data_graph.data.x, None))
            adj0 = self.adj_matrix_leaner.inner_product_all(z)
            alpha = config.ADJ_CUT
            adj0[adj0 < alpha] = 0
            adj1 = adj0

        # adj1 = self.data_graph.get_adj_matrix(self.data_graph.data.edge_index, selfloop=False) # the true adj

        adj1[self.data_graph.train_mask] = 0
        row, col = self.data_graph.train_pos_edge_index
        adj1[row, col] = 1
        adj1[col, row] = 1

        if config.DISTANCE_CONSTRAINT and config.DISTANCE_BORDER_NODE_NUM > 0:
            row, col = self.data_graph.one_hop_edge_index
            adj1[row, col] = 1
            adj1[col, row] = 1
            adj1[self.data_graph.no_edge_mask] = 0
        return adj1


def run(rnd_seed=123):
    config.DISTANCE_CONSTRAINT = True  # distance constrains are used here
    return model.run(RefineAdj, rnd_seed)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(level=logging.INFO, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    config.DISTANCE_BORDER_NODE_NUM = 0
    run(0)

    config.DISTANCE_BORDER_NODE_NUM = 5
    run(0)

