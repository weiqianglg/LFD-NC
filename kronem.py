import math
import os.path as osp
from subprocess import run as process_run
import logging
import numpy as np
import snap
import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = ".\\kronem_bin"


class KronEM:
    def __init__(self, data_graph, observed_graph_name, kronem_exe="kronem.exe"):
        self.data_graph = data_graph
        self.observed_graph_name = observed_graph_name
        self.observed_graph_path = osp.join(BASE_DIR, self.observed_graph_name)
        self.nkroniters = -1
        self.kronem_exe = osp.join(BASE_DIR, kronem_exe)
        self.prob_mtx = None
        self.node_perm = None

    def run_kronem_exe(self):
        self.nkroniters = math.ceil(
            math.log2(self.data_graph.num_nodes))  # c++ (int) ceil(log(double(Nodes)) / log(double(GetDim())));
        g = nx.to_directed(self.data_graph.observed_graph)
        nx.write_edgelist(g, self.observed_graph_path, delimiter="\t", data=False)
        cmd = [self.kronem_exe, f"-i:{self.observed_graph_path}", f"-o:{self.observed_graph_path}",
               f"-d:{self.nkroniters}"]
        logging.info(f"run kronem with args {cmd}")
        process_run(cmd, check=True)
        logging.info("run kronem done")

    def read_mtx(self, mat_path):
        tfin = snap.TFIn(mat_path)
        mtx = snap.TStr(tfin).CStr()
        mtx = mtx[1:-1]  # remove []
        r0, r1 = mtx.split(";")
        r00, r01 = [float(i) for i in r0.split(",")]
        r10, r11 = [float(i) for i in r1.split(",")]
        self.prob_mtx = [[r00, r01], [r10, r11]]
        logging.info(f"read_mtx {self.prob_mtx}")

    def read_nodeperm(self, perm_path):
        tfin = snap.TFIn(perm_path)
        self.node_perm = snap.TIntV(tfin)
        logging.info(f"read_nodeperm {self.node_perm.Len()}")

    def extract_kronem_mtx_nodeperm(self):
        self.read_mtx(f"{self.observed_graph_path}-mtx")
        self.read_nodeperm(f"{self.observed_graph_path}-nodeperm")

    def get_edge_prob(self, NId1, NId2):
        """ C++
        double TKronMtx::GetEdgeProb(int NId1, int NId2, const int& NKronIters) const {
          double Prob = 1.0;
          for (int level = 0; level < NKronIters; level++) {
            Prob *= At(NId1 % MtxDim, NId2 % MtxDim);
            if (Prob == 0.0) { return 0.0; }
            NId1 /= MtxDim;  NId2 /= MtxDim;
          }
          return Prob;
        }
        """
        MtxDim = 2
        Prob = 1.0
        for level in range(self.nkroniters):
            Prob *= self.prob_mtx[NId1 % MtxDim][NId2 % MtxDim]
            if Prob == 0.0:
                return 0.0
            NId1 //= MtxDim
            NId2 //= MtxDim
        return Prob

    def get_kronem_predict_edge_result(self, edge_index):
        row, col = edge_index
        r = [self.get_edge_prob(self.node_perm[row[i].item()], self.node_perm[col[i].item()])
             for i in range(edge_index.size(1))]
        return np.array(r)

    def run(self):
        self.run_kronem_exe()
        self.extract_kronem_mtx_nodeperm()

        pos_edge_index, neg_edge_index = self.data_graph.test_pos_edge_index, self.data_graph.get_test_neg_edge()

        pos_y = np.ones(pos_edge_index.size(1))
        neg_y = np.zeros(neg_edge_index.size(1))
        y = np.hstack([pos_y, neg_y])

        pos_pred = self.get_kronem_predict_edge_result(pos_edge_index)
        neg_pred = self.get_kronem_predict_edge_result(neg_edge_index)
        pred = np.hstack([pos_pred, neg_pred])

        mae_pos = np.sum(pos_y - pos_pred) / len(pos_pred)
        mae_neg = np.sum(neg_pred - neg_y) / len(neg_pred)

        auc, ap, mae = roc_auc_score(y, pred), average_precision_score(y, pred), (mae_pos + mae_neg) / 2

        return (None,  # kronem has no multi metric
                pd.DataFrame([[auc, ap, mae, 'test', '-1']], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']))


def run(rnd_seed=123):
    import config

    def wrap_kronem(data_graph):
        observed_graph_name = f"{config.DATASET}-{rnd_seed}"
        return KronEM(data_graph, observed_graph_name)

    import model
    return model.run(wrap_kronem, rnd_seed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    x, y = run(123)
    print(y)
