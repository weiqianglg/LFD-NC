import os.path as osp
from collections import namedtuple
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikiCS, Actor, WebKB
from torch_geometric.utils import to_undirected
import networkx as nx
import numpy as np

"""all the dataset will return feature tensor and edge_index tensor."""
Data = namedtuple('Data', ['x', 'edge_index'])


def concat_label(data, y):
    if y is None:
        return data
    x = data.x
    unique_y = torch.unique(y)
    unique_y_length = unique_y.size(0)
    code_y = torch.eye(unique_y_length)

    x_with_label = torch.cat((code_y[y], x), dim=1)
    return Data(x=x_with_label, edge_index=data.edge_index)


def Test():
    g = nx.read_edgelist(r"D:\weiqiang\NetworkCompletion\kronem_bin\graph.txt", delimiter='\t', nodetype=int,
                         data=False)
    # g = nx.barabasi_albert_graph(4096, 5)
    g = nx.convert_node_labels_to_integers(g)
    x = torch.eye(g.number_of_nodes())
    edge = [e for e in g.edges]
    edge_index = to_undirected(torch.tensor(edge).transpose(0, 1))
    y1 = torch.zeros(g.number_of_nodes() // 2, dtype=torch.long)
    y2 = torch.ones(g.number_of_nodes() - g.number_of_nodes() // 2, dtype=torch.long)

    return Data(x, edge_index), torch.cat([y1, y2])


def Cora():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y


def Pubmed():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = Planetoid(path, 'Pubmed', transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y


def Citeseer():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = Planetoid(path, 'Citeseer', transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y

def Wikics():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'WikiCS')
    dataset = WikiCS(path, transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y

class MyActor(Actor):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyActor, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

def FilmActor():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'film')

    Actor.raw_file_names = ['out1_node_feature_label.txt', 'out1_graph_edges.txt', "tmp.npz"]

    dataset = Actor(path, transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y

def Cornell():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

    WebKB.raw_file_names = ['out1_node_feature_label.txt', 'out1_graph_edges.txt', "tmp.npz"] # npz is meaningless

    dataset = WebKB(path, 'cornell', transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y

def Texas():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

    WebKB.raw_file_names = ['out1_node_feature_label.txt', 'out1_graph_edges.txt', "tmp.npz"] # npz is meaningless

    dataset = WebKB(path, 'texas', transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y

def Wisconsin():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

    WebKB.raw_file_names = ['out1_node_feature_label.txt', 'out1_graph_edges.txt', "tmp.npz"] # npz is meaningless

    dataset = WebKB(path, 'wisconsin', transform=T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y


if __name__ == '__main__':
    # np.savez("tmp.npz", train_mask=np.array([0]), val_mask=np.array([1]), test_mask=np.array([2]))
    Cornell()
