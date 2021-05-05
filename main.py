import logging
import pandas as pd
import numpy as np
from mlp import run as mlp_run
from gnn import run as gnn_run
from GGCN.run_iso_nodes import ggcn_run
# from kronem import run as kronem_run
from forsati import run as forsati_run
import config


def main():
    # ratio = [0.01 * i for i in range(0, 51)] # sample edge ratio
    # ratio = np.linspace(0, 0.12, 20)
    # ratio = np.logspace(-4, -1, 11) # smb initial strength
    # ratio = np.hstack([[0], ratio])
    ratio = [0, 1, 2, 3, 4, 8, 16, 24, 32] # distance source node number
    all_result = pd.DataFrame()
    for r in ratio:
        # config.TRAIN_SAMPLE_RATIO = r
        # config.SMB_RATIO = r
        config.DISTANCE_BORDER_NODE_NUM = int(r)

        all_best_metric_mlp = pd.DataFrame()
        all_best_metric_gnn = pd.DataFrame()
        all_best_metric_ggcn = pd.DataFrame()
        all_best_metric_kronem = pd.DataFrame()
        all_best_metric_fs = pd.DataFrame()

        for index in range(config.TOTAL_RUN_TIME):
            seed = index
            logging.critical(f"processing {index} round, seed is {seed}")

            if "mpl" in config.ALGO:
                _metric, _best_metric = mlp_run(seed)
            else:
                _best_metric = pd.DataFrame([[0, 0, 0, 'test', -1]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH'])
            all_best_metric_mlp = all_best_metric_mlp.append(_best_metric)

            if "lfd" in config.ALGO:
                _metric, _best_metric = gnn_run(seed)
            else:
                _best_metric = pd.DataFrame([[0, 0, 0, 'test', -1]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH'])
            all_best_metric_gnn = all_best_metric_gnn.append(_best_metric)

            if "ggcn" in config.ALGO:
                _metric, _best_metric = ggcn_run(seed)
            else:
                _best_metric = pd.DataFrame([[0, 0, 0, 'test', -1]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH'])
            all_best_metric_ggcn = all_best_metric_ggcn.append(_best_metric)

            if "kronem" in config.ALGO:
                _metric, _best_metric = kronem_run(seed)
            else:
                _best_metric = pd.DataFrame([[0, 0, 0, 'test', -1]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH'])
            all_best_metric_kronem = all_best_metric_kronem.append(_best_metric)

            if "mc-dt" in config.ALGO:
                _metric, _best_metric = forsati_run(seed)
            else:
                _best_metric = pd.DataFrame([[0, 0, 0, 'test', -1]], columns=['AUC', 'AP', 'MAE', 'TYPE', 'EPOCH'])
            all_best_metric_fs = all_best_metric_fs.append(_best_metric)

        mlp_mean, mlp_std = all_best_metric_mlp.mean(), all_best_metric_mlp.std()
        gnn_mean, gnn_std = all_best_metric_gnn.mean(), all_best_metric_gnn.std()
        ggcn_mean, ggcn_std = all_best_metric_ggcn.mean(), all_best_metric_ggcn.std()
        kronem_mean, kronem_std = all_best_metric_kronem.mean(), all_best_metric_kronem.std()
        fs_mean, fs_std = all_best_metric_fs.mean(), all_best_metric_fs.std()

        ratio_result = pd.DataFrame([(mlp_mean.AUC, mlp_mean.AP, mlp_mean.MAE,
                                      mlp_std.AUC, mlp_std.AP, mlp_std.MAE,
                                      gnn_mean.AUC, gnn_mean.AP, gnn_mean.MAE,
                                      gnn_std.AUC, gnn_std.AP, gnn_std.MAE,
                                      ggcn_mean.AUC, ggcn_mean.AP, ggcn_mean.MAE,
                                      ggcn_std.AUC, ggcn_std.AP, ggcn_std.MAE,
                                      kronem_mean.AUC, kronem_mean.AP, kronem_mean.MAE,
                                      kronem_std.AUC, kronem_std.AP, kronem_std.MAE,
                                      fs_mean.AUC, fs_mean.AP, fs_mean.MAE,
                                      fs_std.AUC, fs_std.AP, fs_std.MAE
                                      )],
                                    columns=['MLP_AUC', 'MLP_AP', 'MLP_MAE',
                                             'MLP_AUC_std', 'MLP_AP_std', 'MLP_MAE_std',
                                             'GNN_AUC', 'GNN_AP', 'GNN_MAE',
                                             'GNN_AUC_std', 'GNN_AP_std', 'GNN_MAE_std',
                                             'GGCN_AUC', 'GGCN_AP', 'GGCN_MAE',
                                             'GGCN_AUC_std', 'GGCN_AP_std', 'GGCN_MAE_std',
                                             'KRONEM_AUC', 'KRONEM_AP', 'KRONEM_MAE',
                                             'KRONEM_AUC_std', 'KRONEM_AP_std', 'KRONEM_MAE_std',
                                             'fs_AUC', 'fs_AP', 'fs_MAE',
                                             'fs_AUC_std', 'fs_AP_std', 'fs_MAE_std'
                                             ])
        logging.critical(f"ratio {r:.4f} done. results :\n{ratio_result.to_string()}")

        all_result = all_result.append(ratio_result)
    all_result.index = ratio
    logging.critical(f"{config.DATASET} final results :\n{ratio_result.to_json()}")
    enable_algo = "_".join(config.ALGO)
    all_result.to_excel(
        f"./data/run_result/{enable_algo}_d{config.DATASET}_l{config.LOSS}_{config.DISTANCE_CONSTRAINT}_b{config.DISTANCE_BORDER_NODE_NUM}_i{config.VAL_INDICATOR}_s{config.SMB_RATIO}_r{config.TRAIN_SAMPLE_RATIO}_R{config.ROUND}.xls")
    return all_result


if __name__ == '__main__':
    logging.basicConfig(filename='./data/dis.log', level=logging.CRITICAL, datefmt="%m-%d-%H-%M-%S", format="%(asctime)s %(message)s", filemode="w")
    # config.DATASET = 'Cora'
    # main()
    # config.DATASET = 'Citeseer'
    # main()
    # config.DATASET = 'Pubmed'
    # main()
    
    config.DATASET = 'Cora'
    config.SMB_RATIO = 0.005
    main()

    config.DATASET = 'Actor'
    config.SMB_RATIO = 0.01
    main()

    config.DATASET = 'Wisconsin'
    config.SMB_RATIO = 0.01
    main()