import logging
import random
from collections import defaultdict
from itertools import combinations, product
import networkx as nx
import numpy as np
import torch

from dataset import Data

from torch_geometric.utils import to_undirected
import config


class SplitGraph(object):
    def __init__(self, data, train_edge_ratio=0.8, val_edge_ratio=0.1, y=None, largest_cc=True):  # y is node label
        """
        :param data: Data in Dataset with x, edge_index
        :param train_edge_ratio: number of positive edges for training
        :param y: data label
        :param largest_cc: only focus on the largest component of the network, distance constrains needed largest_cc=True.
        """
        self.data = data

        self.y = torch.zeros(self.data.x.size(0), dtype=torch.long) if y is None else y
        logging.info("node label class number {:d}".format(torch.unique(self.y).size(0)))

        self.feature_dim = self.data.x.size(1)
        logging.info("graph node feature {:d}".format(self.data.x.size(1)))

        self.graph = self.make_graph(largest_cc=largest_cc)
        self.num_nodes = self.graph.number_of_nodes()

        self.train_edge_ratio = train_edge_ratio
        self.val_edge_ratio = val_edge_ratio

        self.observed_graph = self.get_observed_graph()
        self.observed_index = self.observed_graph.number_of_nodes()

        self.reorder_node()
        # self.store_graph()

        if config.DISTANCE_CONSTRAINT and config.DISTANCE_BORDER_NODE_NUM > 0:
            self.border_node = self.get_border_node(config.DISTANCE_BORDER_NODE_NUM
                                                    )  # for distance constrains, border node is the nodes whose neighbors not all in the observed graph
            self.no_edge_mask, self.one_hop_edge_index = self.get_distance_constrain()

        self.train_mask = None
        self.train_pos_edge_index = self.test_pos_edge_index = self.val_pos_edge_index = None
        self._train_neg_edge_index = self._test_neg_edge_index = None

        self.train_pos_edge_mask = None

        self.split()

    def __relabel_graph(self, x, y, node, index_start=0):
        """helper func for reorder_node"""
        for i, n in enumerate(node, start=index_start):
            x[i] = self.data.x[n]
            y[i] = self.y[n]
            self.graph.nodes[n]["index"] = i
            if i < self.observed_index:
                self.observed_graph.nodes[n]["index"] = i

    def __rebuild_graph(self, g):
        """helper func for reorder_node"""
        _graph = nx.empty_graph()
        _graph.add_nodes_from([g.nodes[u]["index"] for u in g.nodes])
        _graph.add_edges_from([
            (g.nodes[u]["index"], g.nodes[v]["index"]) for u, v in g.edges
        ])
        return _graph

    def reorder_node(self):
        x = torch.zeros(self.num_nodes, self.feature_dim)
        y = torch.zeros(self.num_nodes, dtype=torch.long)

        observed_node = list(self.observed_graph.nodes)
        self.__relabel_graph(x, y, observed_node, 0)
        left_node = set(list(self.graph.nodes)) - set(observed_node)
        self.__relabel_graph(x, y, left_node, self.observed_index)

        self.observed_graph = self.__rebuild_graph(self.observed_graph)
        self.graph = self.__rebuild_graph(self.graph)

        all_edge = [e for e in self.graph.edges]
        self.data = Data(x, torch.tensor(all_edge).transpose(0, 1))
        self.y = y

        logging.info("reorder graph, make observed graph at left-up corner of the adj matrix")

    def store_graph(self, observed_graph_name="observed.edgelist", complete_graph_name="all.edgelist"):
        nx.write_edgelist(nx.to_directed(self.observed_graph), observed_graph_name, data=False, delimiter='\t')
        nx.write_edgelist(nx.to_directed(self.graph), complete_graph_name, data=False, delimiter='\t')
        logging.info(f"write observed graph to {observed_graph_name} and complete graph to {complete_graph_name}")

    def split(self):
        # prepare pos/neg train/test edges
        self.train_mask = self.set_train_mask()
        self.set_pos_edge()
        self.train_pos_edge_mask = self.set_train_pos_edge_mask()
        self.set_neg_edge()

    def make_graph(self, largest_cc=True):
        g = nx.Graph()

        row, col = self.data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        edge_list = [(row[i].item(), col[i].item()) for i in range(row.shape[0])]
        g.add_edges_from(edge_list)

        logging.info("data->graph done. %d nodes with %d edges." % (g.number_of_nodes(), g.number_of_edges()))
        if largest_cc:
            largest_cc = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest_cc)
            logging.info(
                "we only focus on the largest cc. %d nodes with %d edges." % (g.number_of_nodes(), g.number_of_edges()))
        return g

    def get_observed_graph(self):
        """randomly pick a connected subgraph using BFS."""
        train_edge_number = int(self.graph.number_of_edges() * self.train_edge_ratio)
        added_node = set()
        added_edges_number = 0

        _node = list(self.graph.nodes)
        start_node = random.choice(_node)
        added_node.add(start_node)
        logging.debug("random choose start node {}".format(start_node))

        for p, child in nx.bfs_successors(self.graph, start_node):

            for n in child:
                neighbor_n = set(self.graph.neighbors(n))
                n_new_edges_number = len(neighbor_n & added_node)
                added_edges_number += n_new_edges_number
                added_node.add(n)
                if added_edges_number >= train_edge_number:
                    h = self.graph.subgraph(added_node)
                    logging.critical("random sample subgraph done. %d edges sampled. ratio %f, with %d nodes" % (
                        h.number_of_edges(), h.number_of_edges()/self.graph.number_of_edges(), h.number_of_nodes()))
                    return h

        raise RuntimeError("can not get {:d} edges starting from node {:d}".format(train_edge_number, start_node))

    def get_border_node(self, number_limit):
        r = {}
        observed_nodes = set(list(self.observed_graph.nodes))
        for n in self.observed_graph.nodes:
            neighbor_n = set(self.graph.neighbors(n))
            left_edge_number = len(neighbor_n - observed_nodes)
            if left_edge_number > 0:
                r[n] = left_edge_number
        r = sorted(r.items(), key=lambda d: d[1], reverse=True)
        if number_limit > len(r):
            number_limit = len(r)
            logging.debug("border node number limit is too big, no enough candidates, auto adjust it.")
        logging.info(
            f"border node candidates {len(r)}, edge number to unobserved part max {r[0][1]}, min {r[number_limit-1][1]}.")
        return [n for i, (n, _) in enumerate(r) if i < number_limit]

    def get_distance_constrain(self):
        import igraph
        graph_ig = igraph.Graph(n=self.num_nodes, edges=[e for e in self.graph.edges])
        distance = graph_ig.shortest_paths(self.border_node, list(range(self.observed_index, self.num_nodes)))
        border2contour = [None] * len(self.border_node)
        for ib, b in enumerate(self.border_node):
            b_contour = defaultdict(list)  # equal distance nodes
            b_contour[0].append(b)
            for iu, u in enumerate(range(self.observed_index, self.num_nodes)):
                # assert nx.shortest_path_length(self.graph, b, u) == distance[ib][iu]
                b_contour[distance[ib][iu]].append(u)
            border2contour[ib] = b_contour

        logging.info("distances from border nodes to others done.")

        one_hop_index = []  # this position must have edge
        no_edge_mask = np.zeros((self.num_nodes, self.num_nodes),
                                   dtype=np.bool)  # True position can not have edge because distance constrains
        for ib, b in enumerate(self.border_node):
            b_contour = border2contour[ib]
            for d1, d2 in combinations(b_contour.keys(), 2):
                if abs(d2 - d1) > 1:
                    no_edge_mask[np.ix_(b_contour[d1], b_contour[d2])] = True
                    no_edge_mask[np.ix_(b_contour[d2], b_contour[d1])] = True
            for n1 in b_contour[1]:
                one_hop_index.append((b, n1))

        unobserved_nodes_num = self.num_nodes - self.observed_index        
        no_edge_num, one_hop_num  = no_edge_mask.sum()//2, len(one_hop_index)
        all_possible_edge_num = self.observed_index*unobserved_nodes_num + unobserved_nodes_num*(unobserved_nodes_num-1)//2
        certain_rate = (no_edge_num + one_hop_num) / all_possible_edge_num
        logging.critical(f"no edge number {no_edge_num}, one hop edge number {one_hop_num}, uncertain edge number {all_possible_edge_num}, certain ratio {certain_rate:.4f}.")
        return no_edge_mask, torch.tensor(one_hop_index).transpose(0, 1)

    def set_train_mask(self):
        row = col = list(range(self.observed_index))
        train_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        train_mask[0:self.observed_index, 0:self.observed_index] = True
        # for r in row:
        #     train_mask[r, col] = True
        logging.info("set train mask done.")
        return train_mask

    def set_train_pos_edge_mask(self):
        row, col = self.train_pos_edge_index
        pos_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        pos_mask[row, col] = True
        pos_mask[col, row] = True
        return pos_mask

    def get_train_pos_edge_index(self, duplicated=False):
        if not duplicated:
            return self.train_pos_edge_index
        else:
            return to_undirected(self.train_pos_edge_index)

    def set_pos_edge(self):
        train_pos_edge_index = [e for e in self.observed_graph.edges]
        self.train_pos_edge_index = torch.tensor(train_pos_edge_index).transpose(0, 1)
        logging.info("set train pos edge done.")
        test_pos_edge_index, val_pos_edge_index = [], []
        for e in self.graph.edges:
            if e in self.observed_graph.edges:
                continue
            if random.random() * (1 - self.train_edge_ratio) < self.val_edge_ratio:
                val_pos_edge_index.append(e)
            else:
                test_pos_edge_index.append(e)
        self.test_pos_edge_index = torch.tensor(test_pos_edge_index).transpose(0, 1)
        self.val_pos_edge_index = torch.tensor(val_pos_edge_index).transpose(0, 1)
        logging.info("set test pos edge done.")

    def set_neg_edge(self):
        """only set all neg row and col, but leave actual edges later"""
        neg_row, neg_col = self.generate_neg_edge_index(self.train_pos_edge_index, ~self.train_mask)
        self._train_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
        logging.info("set train neg edge done.")

        neg_row, neg_col = self.generate_neg_edge_index(self.test_pos_edge_index, self.train_mask)
        self._test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
        logging.info("set test neg edge done.")

    def generate_neg_edge_index(self, pos_edge_index, fake_pos_mask):
        """generate all neg edges, except pos_edge_index==1, fake_pos_mask==1"""
        row, col = pos_edge_index
        pos_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.uint8)
        pos_mask[row, col] = 1
        pos_mask = pos_mask.triu(diagonal=1)
        pos_mask = pos_mask.to(torch.bool)

        fake_pos_mask = fake_pos_mask.to(torch.uint8)
        fake_pos_mask = fake_pos_mask.triu(diagonal=1)
        fake_pos_mask = fake_pos_mask.to(torch.bool)

        neg_adj_mask = torch.ones(self.num_nodes, self.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[fake_pos_mask] = 0
        neg_adj_mask[pos_mask] = 0
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        return torch.stack([neg_row, neg_col], dim=0)

    def get_neg_edge_helper(self, neg_edge_index, edge_num):
        neg_edge_num = neg_edge_index.size(1)
        edge_num = min(edge_num, neg_edge_num)
        perm = torch.tensor(random.sample(range(neg_edge_num), edge_num))
        perm = perm.to(torch.long)
        r = neg_edge_index[:, perm]
        neg_edge_index = r
        return neg_edge_index

    def get_train_neg_edge(self):
        return self.get_neg_edge_helper(self._train_neg_edge_index,
                                        self.observed_graph.number_of_edges())

    def get_test_neg_edge(self):
        return self.get_neg_edge_helper(self._test_neg_edge_index,
                                        self.test_pos_edge_index.size(1))

    def get_val_neg_edge(self):
        """self._test_neg_edge_index is all the unobserved neg edges, val and test are both generated randomly from it."""
        return self.get_neg_edge_helper(self._test_neg_edge_index,
                                        self.val_pos_edge_index.size(1))

    def get_train_neg_edge_by_semi_mining(self, distance_z):
        import time
        ts = time.perf_counter()
        distance_z = distance_z[:self.observed_index, :self.observed_index]
        result = []
        sorted_dis, index_dis = torch.sort(distance_z, descending=True)
        for i, row in enumerate(index_dis):
            positive_count = 0
            pos_mask = self.train_pos_edge_mask[i]
            for j in row:
                j = j.item()
                if pos_mask[j]:
                    positive_count += 1
                elif positive_count > 0:
                    result.extend([(i, j)] * positive_count)
                    positive_count = 0
        result = torch.tensor(result).t()
        te = time.perf_counter()
        print("get_train_neg_edge_by_semi_mining", te - ts)
        return result.to(distance_z.device)

    def get_train_neg_edge_by_hard_mining(self, distance_z):
        import time
        ts = time.perf_counter()
        distance_z = distance_z[:self.observed_index, :self.observed_index]
        result = []
        sorted_dis, index_dis = torch.sort(distance_z, descending=True)
        for i, row in enumerate(index_dis):
            positive_count = 0
            pos_mask = self.train_pos_edge_mask[i]
            for j in row:
                j = j.item()
                if pos_mask[j]:
                    positive_count += 1
                elif positive_count > 0:
                    result.extend([(i, j)] * positive_count)
                    positive_count = 0
        result = torch.tensor(result).t()
        te = time.perf_counter()
        print("get_train_neg_edge_by_semi_mining", te - ts)
        return result.to(distance_z.device)

    def to(self, dev):
        self.data = Data(self.data.x.to(dev), self.data.edge_index.to(dev))
        self.train_mask = self.train_mask.to(dev)
        self.train_pos_edge_mask = self.train_pos_edge_mask.to(dev)
        self.train_pos_edge_index = self.train_pos_edge_index.to(dev)
        self.test_pos_edge_index = self.test_pos_edge_index.to(dev)
        self._train_neg_edge_index = self._train_neg_edge_index.to(dev)
        self._test_neg_edge_index = self._test_neg_edge_index.to(dev)

    def save_npz(self, filename, **kwargs):
        datax = self.data.x.numpy()
        edge_index = self.data.edge_index.numpy()
        observed_index = np.array(self.observed_index)
        kwargs.update({'datax': datax, 'edge_index': edge_index,
                       'observed_index': observed_index})
        np.savez(filename, **kwargs)

    def get_adj_matrix(self, size, edge_index, selfloop=False):
        """construct adj matrix by size x size, and edge_index will be set 1 """
        adj = torch.zeros(size, size)
        row, col = edge_index
        adj[row, col] = 1
        adj[col, row] = 1
        if selfloop:
            adj.add_(torch.eye(size))
        return adj

    def get_label_smb_matrix(self):
        y = self.y
        uy = torch.unique(y)
        e = torch.eye(uy.size(0))
        l = e[y]
        return l.mm(l.t())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    from dataset import Cora, Test

    random.seed(12345)
    data, _ = Test()
    dg = SplitGraph(data, 0.75)

    # distance_z = torch.rand(dg.num_nodes, dg.num_nodes)
    # for i in range(10):
    #     dg.get_train_neg_edge_by_semi_mining(distance_z)
