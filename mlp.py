import logging
import torch.nn as nn
import pandas as pd
import config
import model


# This is a basic multilayer perceptron
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=True):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size, bias=False))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        z = self.net(x)
        return z


class SimilarityMetric(model.ModelBase):
    def __init__(self, data_graph):
        super().__init__(data_graph)
        self.neural_network = MLP([data_graph.feature_dim, config.HIDDEN_DIM, config.OUT_DIM])

    def run(self):
        return super().run(self.data_graph.data.x)


def run(rnd_seed=123):
    config.DISTANCE_CONSTRAINT = False  # distance constrains are not used here
    return model.run(SimilarityMetric, rnd_seed)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    # main('Citeseer')
    all_best_metric = pd.DataFrame()
    for seed in range(10):
        _metric, _best_metric = run(seed)
        all_best_metric = all_best_metric.append(_best_metric)

    print(all_best_metric.mean())
    print(all_best_metric.std())
