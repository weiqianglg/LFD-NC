# GENERATIVE GRAPH CONVOLUTIONAL NETWORK FOR GROWING GRAPHS (ICASSP 2019)
# Authors: Da Xu*, Chuanwei Ruan*, Kamiya Motwani, Sushant Kumar, Evren Korpeoglu, Kannan Achan

import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
import model
import config
from GGCN.layers import RecursiveGraphConvolutionStepAddOn
from GGCN.graph_data import GraphSequenceBfsRandSampler
from GGCN.loss import recursive_loss_with_noise
from GGCN.utils import sample_reconstruction

# Don not use this, some place is wrong, use run_iso_nodes.py in GGCN
class CompletionWithGGcn(model.ModelBase):
    def __init__(self, data_graph):
        super().__init__(data_graph)
        self.neural_network = RecursiveGraphConvolutionStepAddOn(data_graph.feature_dim, config.HIDDEN_DIM,
                                                                 config.OUT_DIM, dropout=0.0)
        self.adj = self.feat = None
        self.size_update = int(self.data_graph.num_nodes * 0.33 * config.TRAIN_SAMPLE_RATIO)

    def train(self):
        self.optimizer.zero_grad()
        self.neural_network.train()
        loss = recursive_loss_with_noise(self.neural_network, self.adj, self.feat, self.size_update, None)
        loss.backward()
        self.loss_val = loss
        self.optimizer.step()

    def inner_product(self, z, edge_index):
        """here z is a matrix return by sample_reconstruction, overwrite the base method for test"""
        row, col = edge_index
        return z[row, col].sigmoid()

    def eval(self, g_adj, pos_edge_index, neg_edge_index):
        assert pos_edge_index.size(1) == neg_edge_index.size(1)
        self.neural_network.eval()
        with torch.no_grad():
            z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = self.neural_network(torch.from_numpy(g_adj),
                                                                                       self.feat,
                                                                                       self.data_graph.data.x[self.data_graph.observed_index:, :])
            z_mean = torch.cat((z_mean_old, z_mean_new))
            z_log_std = torch.cat((z_log_std_old, z_log_std_new))

            adj_h = sample_reconstruction(z_mean, z_log_std)
        return self.test(adj_h, pos_edge_index, neg_edge_index)

    def run(self):
        val_metric, test_metric = self.run_prepare()

        g_adj = self.data_graph.get_adj_matrix(self.data_graph.observed_index,
                                               self.data_graph.train_pos_edge_index, selfloop=True).detach().cpu().numpy()
        X = self.data_graph.data.x[:self.data_graph.observed_index, :].detach().cpu().numpy()

        dataset = GraphSequenceBfsRandSampler(g_adj, X, num_permutation=config.EPOCH, seed=123, fix=False)
        params = {'batch_size': 1,
                  'shuffle': True,
                  'num_workers': 1}
        dataloader = torch.utils.data.DataLoader(dataset, **params)

        for epoch, (adj, feat) in enumerate(dataloader):
            self.adj = adj[0]
            self.feat = feat[0]
            if self.adj.size()[0] <= self.size_update:
                print("sample size {} too small, skipped!".format(adj.size()[0]))
                continue

            self.train()
            self.record_metric(epoch, g_adj, val_metric, test_metric)

            if epoch - self.best_epoch > config.STOP_DELTA_EPOCH:
                logging.info(f"metric indicator {self.val_indicator_index} \
                            can not be improved in {config.STOP_DELTA_EPOCH} epoch, stop")
                break
        logging.critical(
            f"best epoch {self.best_epoch}. test auc {test_metric[self.best_epoch][0]:.4f}, \
            ap {test_metric[self.best_epoch][1]:.4f}, mae {test_metric[self.best_epoch][2]:.4f}")

        return (pd.DataFrame(val_metric + test_metric, columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']),
                pd.DataFrame([test_metric[self.best_epoch]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH']))


def run(rnd_seed=0):
    config.DISTANCE_CONSTRAINT = False  # distance constrains are not used here
    from multiprocessing import freeze_support
    freeze_support()
    return model.run(CompletionWithGGcn, rnd_seed)

    from GGCN.run_iso_nodes import ggcn_run
    return ggcn_run(rnd_seed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    metric, best_metric = run()
    print(best_metric)
