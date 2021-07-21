import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import config


class ModelBase:
    def __init__(self, data_graph):
        self.data_graph = data_graph
        self.in_channels = data_graph.feature_dim
        self.out_channels = config.OUT_DIM
        self.neural_network = None
        self.optimizer = None
        self.loss_val = 0.0
        self.best_epoch = -1
        self.val_indicator_index = {'AUC': 0, 'AP': 1, 'MAE': 2}[config.VAL_INDICATOR]
        self.max_val_metric = 0

    def inner_product(self, z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def inner_product_all(self, z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)

    def loss(self, z, pos_edge_index, neg_edge_index):
        if config.LOSS == 'binary_cross_entropy':
            eps = 1e-15
            pos_loss = -torch.log(
                self.inner_product(z, pos_edge_index) + eps).mean()
            neg_loss = -torch.log(
                1 - self.inner_product(z, neg_edge_index) + eps).mean()
        elif config.LOSS == 'F_norm':
            pos_loss = torch.norm(1 - self.inner_product(z, pos_edge_index))
            neg_loss = torch.norm(self.inner_product(z, neg_edge_index))
        else:
            raise ValueError(f"Parametor.LOSS meet unknown value {config.LOSS}.")
        return pos_loss + neg_loss

    def train(self, x):
        self.optimizer.zero_grad()
        self.neural_network.train()

        z = self.neural_network(x)
        train_neg_edge = self.data_graph.get_train_neg_edge()
        loss = self.loss(z,
                         self.data_graph.train_pos_edge_index,
                         train_neg_edge)
        loss.backward(retain_graph=False)
        self.loss_val = loss
        self.optimizer.step()

    def test(self, z, pos_edge_index, neg_edge_index):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.inner_product(z, pos_edge_index)
        neg_pred = self.inner_product(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        pos_pred = pos_pred.detach().cpu().numpy()
        neg_pred = neg_pred.detach().cpu().numpy()
        mae_pos = np.sum(np.ones(len(pos_pred)) - pos_pred) / len(pos_pred)
        mae_neg = np.sum(neg_pred - np.zeros(len(neg_pred))) / len(neg_pred)

        return roc_auc_score(y, pred), average_precision_score(y, pred), (mae_pos + mae_neg) / 2

    def eval(self, x, pos_edge_index, neg_edge_index):
        assert pos_edge_index.size(1) == neg_edge_index.size(1)
        self.neural_network.eval()
        with torch.no_grad():
            z = self.neural_network(x)
        return self.test(z, pos_edge_index, neg_edge_index)

    def run_prepare(self):
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=config.LR, weight_decay=0.0)  # 5e-6
        val_metric, test_metric = [], []
        self.best_epoch = config.WARM_EPOCH
        return val_metric, test_metric

    def record_metric(self, epoch, x, val_metric, test_metric):
        auc, ap, mae = self.eval(x,
                                 self.data_graph.val_pos_edge_index,
                                 self.data_graph.get_val_neg_edge())
        val_metric.append([auc, ap, mae, 'val', epoch])
        if val_metric[-1][self.val_indicator_index] >= self.max_val_metric and epoch > config.WARM_EPOCH:
            self.max_val_metric = val_metric[-1][self.val_indicator_index]
            self.best_epoch = epoch

        logging.info(
            f'epoch: {epoch:03d}, val, auc {auc:.4f}, ap {ap:.4f}, mae {mae:.4f}')

        auc, ap, mae = self.eval(x,
                                 self.data_graph.test_pos_edge_index,
                                 self.data_graph.get_test_neg_edge())
        test_metric.append([auc, ap, mae, 'test', epoch])
        logging.info(
            f'epoch: {epoch:03d}, test, auc {auc:.4f}, ap {ap:.4f}, mae {mae:.4f}')

    def run(self, x):
        val_metric, test_metric = self.run_prepare()
        for epoch in range(config.EPOCH):
            self.train(x)
            if epoch % 10 == 0:
                self.record_metric(epoch, x, val_metric, test_metric)

            if epoch - self.best_epoch > config.STOP_DELTA_EPOCH:
                logging.info(f"metric indicator {self.val_indicator_index} \
                can not be improved in {config.STOP_DELTA_EPOCH} epoch, stop")
                break
        row_index = self.best_epoch // 10
        logging.critical(
            f"best epoch {self.best_epoch}. test auc {test_metric[row_index][0]:.4f}, \
            ap {test_metric[row_index][1]:.4f}, mae {test_metric[row_index][2]:.4f}")

        return (pd.DataFrame(val_metric + test_metric, columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']),
                pd.DataFrame([test_metric[row_index]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']))


def run(predictor, rnd_seed=123):
    import random

    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    ratio = config.TRAIN_SAMPLE_RATIO
    from dataset import Test, Pubmed, Cora, Citeseer, Wikics, FilmActor, Cornell, Texas, Wisconsin, concat_label
    dataset = {"Pubmed": Pubmed, "Cora": Cora, "Citeseer": Citeseer, "WikiCS": Wikics, "Actor": FilmActor,
               "Cornell": Cornell, "Texas": Texas, "Wisconsin": Wisconsin}
    from data_graph import SplitGraph

    data, y = dataset[config.DATASET]()
    data = concat_label(data, y)
    data_graph = SplitGraph(data, train_edge_ratio=ratio, val_edge_ratio=(1 - ratio) / 2.0, y=y)
    fg = predictor(data_graph)
    metric, best_metric = fg.run()
    return metric, best_metric
