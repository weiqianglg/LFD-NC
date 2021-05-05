EPOCH = 500  # total epoch number
ROUND = 4  # total refine round number
SMB_RATIO = 0.00  # label community ratio for graph adj matrix
ADJ_CUT = 0.5  # value lower than this will be 0
DISTANCE_CONSTRAINT = True  # use distance constraint or not
DISTANCE_BORDER_NODE_NUM = 50  # number of border nodes for distances observation
DATASET = 'Wisconsin'  # Test, Pubmed, Cora, Citeseer, Wikics, Actor, Cornell, Texas, Wisconsin
TRAIN_SAMPLE_RATIO = 0.5  # sample edge ratio for train
LOSS = 'binary_cross_entropy'  # loss function, option contains 'binary_cross_entropy', 'F_norm'
VAL_INDICATOR = 'AUC'  # validation indicator, we pick the best for val, option contains 'AUC', 'AP', 'MAE'
WARM_EPOCH = 100  # before pick the vest val, we first warm up some epoch
STOP_DELTA_EPOCH = 20  # if current epoch > best epoch + STOP_DELTA_EPOCH, we stop training
LR = 0.01  # learning rate
OUT_DIM = 32
HIDDEN_DIM = 64
TOTAL_RUN_TIME = 5  # we run this times for results' mean
ALGO =["lfd"] #["mpl", "ggcn", "lfd", "mc-dt", "kronem"]#

def get_parameters():
    p = globals()
    for k in list(p.keys()):
        if not k[0].isupper():
            del p[k]
    return p


if __name__ == '__main__':
    import pprint

    pprint.pprint(get_parameters())
