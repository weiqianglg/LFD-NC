import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import config
import model


class SBM:
    def __init__(self, data_graph):
        self.data_graph = data_graph
        self.sbm_mat = self.data_graph.get_sbm_matrix()

    def test(self, pos_edge_index, neg_edge_index):

        pos_y = np.ones(pos_edge_index.size(1))
        neg_y = np.zeros(neg_edge_index.size(1))
        y = np.hstack([pos_y, neg_y])

        pos_pred = self.sbm_mat[pos_edge_index[0], pos_edge_index[1]]
        neg_pred = self.sbm_mat[neg_edge_index[0], pos_edge_index[1]]
        pred = np.hstack([pos_pred, neg_pred])

        mae_pos = np.sum(np.ones(len(pos_pred)) - pos_pred) / len(pos_pred)
        mae_neg = np.sum(neg_pred - np.zeros(len(neg_pred))) / len(neg_pred)

        return roc_auc_score(y, pred), average_precision_score(y, pred), (mae_pos + mae_neg) / 2

    def run(self):
        auc, ap, mae = self.test(self.data_graph.test_pos_edge_index, self.data_graph.get_test_neg_edge())
        return (pd.DataFrame([], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']),
                pd.DataFrame([[auc, ap, mae, 'test', 0]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']))


def run(rnd_seed=123):
    config.DISTANCE_CONSTRAINT = False  # distance constrains are not used here
    return model.run(SBM, rnd_seed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    datasets = ["Cora", "Pubmed", "Citeseer", "WikiCS", "Actor", "Cornell", "Texas", "Wisconsin"]
    for dataset in datasets:
        config.DATASET = dataset
        all_best_metric = pd.DataFrame()
        for seed in range(10):
            _metric, _best_metric = run(seed)
            all_best_metric = all_best_metric.append(_best_metric)
        print(dataset)
        print(all_best_metric.mean())
        print(all_best_metric.std())
