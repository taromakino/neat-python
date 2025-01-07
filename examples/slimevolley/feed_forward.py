import neat
import numpy as np
from neat.activations import *


NEAT_TO_NUMPY_ACTIVATION = {
    identity_activation: lambda x: x,
    relu_activation: lambda x: np.maximum(0, x),
    sin_activation: lambda x: np.sin(x),
    square_activation: lambda x: np.square(x),
    sigmoid_activation: lambda x: 1 / (1 + np.exp(-x)),
    tanh_activation: lambda x: np.tanh(x),
}


class BatchFeedForwardNetwork(neat.nn.FeedForwardNetwork):
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        batch_size = len(inputs)

        # The input node values are given to us
        node_to_values = {}
        for i, input_node in enumerate(self.input_nodes):
            node_to_values[input_node] = inputs[:, i]

        # node_evals is sorted in topological order
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            # We compute the value of each node by taking a weighted sum of the node's parent values, multiplying it
            # by the response, adding a bias, and applying an activation function
            weighted_sum = 0.
            for parent_node, weight in links:
                weighted_sum += node_to_values[parent_node] * weight

            # act_func is a neat-python activation function, we use the dict to get the numpy version
            node_to_values[node] = NEAT_TO_NUMPY_ACTIVATION[act_func](bias + response * weighted_sum)

        outputs = []
        for output_node in self.output_nodes:
            if output_node in node_to_values:
                outputs.append(node_to_values[output_node])
            else:
                # If an output node has no parent nodes, then set its value to zero
                outputs.append(np.zeros(batch_size))
        outputs = np.stack(outputs, axis=1)

        return outputs

    @staticmethod
    def create(genome, config):
        superclass = neat.nn.FeedForwardNetwork.create(genome, config)
        return BatchFeedForwardNetwork(superclass.input_nodes, superclass.output_nodes, superclass.node_evals)