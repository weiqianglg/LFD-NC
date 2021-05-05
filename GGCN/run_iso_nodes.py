import argparse
import numpy as np
import torch
from GGCN.layers import RecursiveGraphConvolutionStepAddOn
from GGCN.graph_data import GraphSequenceBfsRandSampler
from GGCN.loss import recursive_loss_with_noise
from GGCN.utils import sample_reconstruction
import config
import logging


def read_citation_dat(ratio=0.3, rnd_seed=0):
    '''
    dataset: {'cora', 'citeseer', 'pubmed'}
    '''
    import random

    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    from dataset import Test, Pubmed, Cora, Citeseer, Wikics, FilmActor, Cornell, Texas, Wsconsin, concat_label
    dataset = {"Pubmed": Pubmed, "Cora": Cora, "Citeseer": Citeseer, "WikiCS": Wikics, "Actor": FilmActor,
               "Cornell": Cornell, "Texas": Texas, "Wsconsin": Wsconsin}
    from data_graph import SplitGraph

    data, y = dataset[config.DATASET]()
    data = concat_label(data, y)
    data_graph = SplitGraph(data, train_edge_ratio=ratio, val_edge_ratio=(1 - ratio) / 2.0, y=y)
    adj_select = data_graph.get_adj_matrix(data_graph.num_nodes,
                                           data_graph.data.edge_index, selfloop=True).detach().cpu().numpy()
    X_select = data_graph.data.x.detach().cpu().numpy()

    cut_idx = data_graph.observed_index
    adj_train = adj_select[:cut_idx, :cut_idx]
    X_train = X_select[:cut_idx, :]
    return adj_train, X_train, adj_select, X_select, data_graph


def ggcn_predict(z, edge_index):
    """here z is a matrix return by sample_reconstruction, overwrite the base method for test"""
    row, col = edge_index
    return z[row, col].sigmoid()


def get_score(z, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = ggcn_predict(z, pos_edge_index)
    neg_pred = ggcn_predict(z, neg_edge_index)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    pred[np.isinf(pred)] = 0.0

    pos_pred = pos_pred.detach().cpu().numpy()
    neg_pred = neg_pred.detach().cpu().numpy()
    mae_pos = np.sum(np.ones(len(pos_pred)) - pos_pred) / len(pos_pred)
    mae_neg = np.sum(neg_pred - np.zeros(len(neg_pred))) / len(neg_pred)
    from sklearn.metrics import roc_auc_score, average_precision_score
    return roc_auc_score(y, pred), average_precision_score(y, pred), (mae_pos + mae_neg) / 2


def ggcn_run(rnd_seed):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=config.HIDDEN_DIM)
    parser.add_argument('--out_dim', type=int, default=config.OUT_DIM)
    parser.add_argument('--update_ratio', type=float, default=0.33)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--refit', type=int, default=0)
    parser.add_argument('--permute', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=config.TRAIN_SAMPLE_RATIO)
    args = parser.parse_args()

    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    train_ratio = args.train_ratio

    g_adj, X, g_adj_all, X_all, data_graph = read_citation_dat(ratio=train_ratio)
    num_nodes = g_adj_all.shape[0]
    num_edges = ((g_adj_all > 0).sum() - num_nodes) / 2
    print([num_nodes, num_edges])

    size_update = int(num_nodes * args.update_ratio * train_ratio)

    seed = float(args.seed) if args.seed else None
    unseen = True
    refit = args.refit > 0
    permute = args.permute > 0

    norm = None
    special = 'nodropout_DEBUG'

    filename = '_'.join(['equal_size_cite', special, config.DATASET,
                         'size', str(size_update),
                         'hidden', str(hidden_dim),
                         'out', str(out_dim),
                         'fix', str(seed is not None),
                         'unseen', str(unseen),
                         'refit', str(refit),
                         'norm', str(norm),
                         'permute', str(permute),
                         'seed', str(seed)])
    filename = './data/' + filename

    logging.basicConfig(level=logging.DEBUG, filename=filename,
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    if seed is not None:
        np.random.seed(seed)

    features_dim = X.shape[1]

    dataset = GraphSequenceBfsRandSampler(g_adj, X, num_permutation=400, seed=seed, fix=False)

    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1}

    dataloader = torch.utils.data.DataLoader(dataset, **params)

    # gcn_step = RecursiveGraphConvolutionStep(features_dim, hidden_dim, out_dim)
    gcn_step = RecursiveGraphConvolutionStepAddOn(features_dim, hidden_dim, out_dim, dropout=0.0)

    optimizer = torch.optim.Adam(gcn_step.parameters(), lr=0.01)


    best_epoch = config.WARM_EPOCH
    max_val_metric = 0
    for batch_idx, (adj, feat) in enumerate(dataloader):
        adj = adj[0]
        feat = feat[0]

        if adj.size()[0] <= size_update:
            print("sample size {} too small, skipped!".format(adj.size()[0]))
            continue

        # train R-GCN

        optimizer.zero_grad()
        gcn_step.train()
        # loss = recursive_loss(gcn_step, adj, feat, size_update)
        loss = recursive_loss_with_noise(gcn_step, adj, feat, size_update, norm)
        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:
            info = 'R-GCN [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx, len(dataloader),
                                                                  100. * batch_idx / len(dataloader),
                                                                  loss.item())
            print(info)
            logging.info(info)

            with torch.no_grad():
                adj_truth_all = torch.from_numpy(g_adj_all.astype(np.float32))

                feat = torch.from_numpy(X)
                feat_all = torch.from_numpy(X_all)

                # # test r-gcn
                gcn_step.eval()
                z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(torch.from_numpy(g_adj), feat,
                                                                                feat_all[feat.size()[0]:, :])
                z_mean = torch.cat((z_mean_old, z_mean_new))
                z_log_std = torch.cat((z_log_std_old, z_log_std_new))

                adj_h = sample_reconstruction(z_mean, z_log_std)
                if refit:
                    adj_hat = (adj_h > 0).type(torch.FloatTensor)
                    adj_hat[:feat.size(0), :feat.size(0)] = torch.from_numpy(g_adj)
                    z_mean, z_log = gcn_step(adj_hat, feat_all)
                    adj_h = sample_reconstruction(z_mean, z_log_std)

                test_metric = get_score(adj_h, data_graph.val_pos_edge_index,
                                        data_graph.get_val_neg_edge())
                print("eval metric ", test_metric)
                ind = {'AUC': 0, 'AP': 1, 'MAE': 2}[config.VAL_INDICATOR]
                if test_metric[ind] > max_val_metric and batch_idx > config.WARM_EPOCH:
                    max_val_metric = test_metric[ind]
                    best_epoch = batch_idx

                    auc_rgcn_v, ap_rgcn_v, mae_rgcn_v = get_score(adj_h, data_graph.test_pos_edge_index,
                                                            data_graph.get_test_neg_edge())
                    print("test metric ", [auc_rgcn_v, ap_rgcn_v, mae_rgcn_v])

        if batch_idx - best_epoch > config.STOP_DELTA_EPOCH:
            logging.info(f"metric indicator {ind} \
            can not be improved in {config.STOP_DELTA_EPOCH} epoch, stop")
            break
    logging.critical(
        f"best epoch {best_epoch}. test auc {auc_rgcn_v:.4f}, \
        ap {ap_rgcn_v:.4f}, mae {mae_rgcn_v:.4f}")
    import pandas as pd
    return None, pd.DataFrame([[auc_rgcn_v, ap_rgcn_v, mae_rgcn_v, 'test', best_epoch]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt="%m-%d-%H-%M-%S", format="%(asctime)s %(message)s")
    from multiprocessing import freeze_support

    freeze_support()
    config.TRAIN_SAMPLE_RATIO = 0.2
    ggcn_run(0)
    config.TRAIN_SAMPLE_RATIO = 0.8
    ggcn_run(0)
