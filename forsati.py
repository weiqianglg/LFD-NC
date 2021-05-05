import logging
import random
import networkx as nx
import numpy as np
import scipy.linalg
import sklearn.metrics.pairwise as pdist
from sklearn.metrics import roc_auc_score, average_precision_score
import config
import model
import pandas as pd


class Forsati(object):
    def __init__(self, data_graph):
        self.data_graph = data_graph
        self.observed_index = data_graph.observed_index
        self.graph_matrix = self.get_graph_matrix()
        self.similarity_matrix = self.get_feature_similarity_matrix(self.data_graph.data.x)
        # self.similarity_matrix = self.graph_matrix + np.eye(self.data_graph.num_nodes)

    def get_graph_matrix(self):
        graph_matrix = nx.adjacency_matrix(self.data_graph.graph, nodelist=range(self.data_graph.num_nodes)).todense()
        logging.info("graph matrix done.")
        return np.array(graph_matrix)

    @staticmethod
    def get_feature_similarity_matrix(data_x):
        feature_similarity = pdist.pairwise_distances(data_x, metric="cosine", n_jobs=8)
        feature_similarity = 1 - feature_similarity  # / np.max(feature_similarity)
        logging.info("feature similarity matrix done.")
        return feature_similarity

    def get_Us(self, top_s):
        e_vals, e_vecs = scipy.linalg.eigh(self.similarity_matrix)
        sorted_indices = np.argsort(e_vals)
        logging.info("eig done.")
        Us = e_vecs[:, sorted_indices[:-top_s - 1:-1]]
        return Us

    def get_U_s(self, Us):
        row = random.sample(range(self.data_graph.num_nodes), self.observed_index)
        row = list(range(self.observed_index))
        return Us[row, :]

    def get_O(self):
        return self.graph_matrix[0:self.observed_index, 0:self.observed_index]

    def get_A_h(self, top_s=None):
        # np.save("feature_similarity_matrix.npy", self.feature_similarity_matrix)
        if top_s is None:
            top_s = 20  # self.graph_matrix.ndim + 8
            logging.info(f"set top_s as rank of graph matrix, {top_s}")
        Us = self.get_Us(top_s)
        U_s = self.get_U_s(Us)

        # pinv = np.linalg.pinv(U_s.T @ U_s, rcond=1e-40)
        pinv = scipy.linalg.pinvh(U_s.T @ U_s, check_finite=False)

        # y = np.allclose(pinv, pinv.T, atol=0.01)
        # print("is close.", y)
        A_ = pinv @ U_s.T @ self.get_O() @ U_s @ pinv

        return Us @ A_ @ Us.T

    def get_scores(self, A_):
        A_c = A_.copy().real
        A_c[A_c < 0] = 0
        A_c[A_c > 1] = 1

        row, col = self.data_graph.test_pos_edge_index
        pos_edge_pred = A_c[row, col]

        row, col = self.data_graph.get_test_neg_edge()
        neg_edge_pred = A_c[row, col]

        testY = np.hstack((np.ones(len(pos_edge_pred)), np.zeros(len(neg_edge_pred))))
        pred = np.hstack((pos_edge_pred, neg_edge_pred))

        mae_pos = np.sum(np.ones(len(pos_edge_pred)) - pos_edge_pred) / len(pos_edge_pred)
        mae_neg = np.sum(neg_edge_pred - np.zeros(len(neg_edge_pred))) / len(neg_edge_pred)

        return roc_auc_score(testY, pred), average_precision_score(testY, pred), (mae_pos+mae_neg)/2

    def run(self):
        A_ = self.get_A_h()
        logging.info("A_ max %f, min %f, mean %f." % (A_.max(), A_.min(), A_.mean()))
        auc, ap, mae = self.get_scores(A_)

        return (None,
                pd.DataFrame([[auc, ap, mae, 'test', -1]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']))


def run(rnd_seed=123):

    config.DISTANCE_CONSTRAINT = False  # distance constrains are not used here
    return model.run(Forsati, rnd_seed)





def main():
    ratio = [0.1 * i for i in range(1, 10)]

    all_result = pd.DataFrame()
    for r in ratio:
        config.TRAIN_SAMPLE_RATIO = r

        all_best_metric_fs = pd.DataFrame()

        for index in range(config.TOTAL_RUN_TIME):
            seed = index
            logging.critical(f"processing {index} round, seed is {seed}")

            _metric, _best_metric = run(seed)
            all_best_metric_fs = all_best_metric_fs.append(_best_metric)

        fs_mean, fs_std = all_best_metric_fs.mean(), all_best_metric_fs.std()

        ratio_result = pd.DataFrame([(fs_mean.AUC, fs_mean.AP, fs_mean.MAE,
                                      fs_std.AUC, fs_std.AP, fs_std.MAE
                                      )],
                                    columns=['fs_AUC', 'fs_AP', 'fs_MAE',
                                             'fs_AUC_std', 'fs_AP_std', 'fs_MAE_std'
                                             ])
        logging.critical(f"ratio {r:.2f} done. results :\n{ratio_result.to_string()}")

        all_result = all_result.append(ratio_result)
    all_result.index = ratio
    logging.critical(f"{config.DATASET} final results :\n{ratio_result.to_json()}")
    all_result.to_excel(
        f"./data/forsati_{config.DATASET}_{config.LOSS}_{config.DISTANCE_BORDER_NODE_NUM}_ind{config.VAL_INDICATOR}_{config.SMB_RATIO}.xls")
    return all_result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt="%m-%d-%H-%M-%S", format="%(asctime)s %(message)s")
    config.DATASET = 'Cornell'
    config.SMB_RATIO = 0.01
    main()