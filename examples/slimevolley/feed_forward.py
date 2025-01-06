import neat
import numpy as np
from neat.activations import *


ACTIVATION_DICT = {
    identity_activation: lambda x: x,
    relu_activation: lambda x: np.maximum(0, x),
    sigmoid_activation: lambda x: 1 / (1 + np.exp(-x)),
    tanh_activation: lambda x: np.tanh(x),
}


class FeedForwardNetwork(neat.nn.FeedForwardNetwork):
    def activate(self, inputs):
        batch_size = len(inputs)
        values = {}
        for i, input_node in enumerate(self.input_nodes):
            values[input_node] = inputs[:, i]

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            weighted_sum = 0.
            for parent_node, weight in links:
                weighted_sum += values[parent_node] * weight
            values[node] = ACTIVATION_DICT[act_func](bias + response * weighted_sum)

        outputs = []
        for output_node in self.output_nodes:
            if output_node in values:
                outputs.append(values[output_node])
            else:
                outputs.append(np.zeros(batch_size))
        outputs = np.stack(outputs, axis=1)

        return outputs

    @staticmethod
    def create(genome, config):
        superclass = neat.nn.FeedForwardNetwork.create(genome, config)
        return FeedForwardNetwork(superclass.input_nodes, superclass.output_nodes, superclass.node_evals)