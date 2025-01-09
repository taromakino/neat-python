import graphviz
import os
import pickle
from neat import Config
from neat.genome import DefaultGenome
from neat.reporting import BaseReporter


def save_file(path, obj):
    dpath = os.path.dirname(path)
    os.makedirs(dpath, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


class VisualizeReporter(BaseReporter):
    def __init__(self, out_dir, population):
        self.out_dir = out_dir
        self.population = population

    def visualize(
            self,
            out_dir: str,
            config: Config,
            genome: DefaultGenome,
            generation: int,
    ) -> None:
        node_names, node_colors = {}, {}

        node_attrs = {
            "shape": "circle",
            "fontsize": "9",
            "height": "0.2",
            "width": "0.2"}

        dot = graphviz.Digraph(node_attr=node_attrs)

        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {"style": "filled", "shape": "box", "fillcolor": node_colors.get(k, "lightgray")}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            # Get the node gene to access its activation function for output nodes
            node_gene = genome.nodes[k]
            activation = node_gene.activation if hasattr(node_gene, "activation") else "unknown"

            # Create label with both node number and activation function
            label = f"{name}\n{activation}"

            node_attrs = {"style": "filled",
                          "fillcolor": node_colors.get(k, "lightblue"),
                          "label": label}
            dot.node(name, _attributes=node_attrs)

        used_nodes = set(genome.nodes.keys())
        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            # Get the node gene to access its activation function
            node_gene = genome.nodes[n]
            activation = node_gene.activation if hasattr(node_gene, "activation") else "unknown"

            # Create label with both node number and activation function
            label = f"{n}\n{activation}"

            attrs = {"style": "filled",
                     "fillcolor": node_colors.get(n, "white"),
                     "label": label}
            dot.node(str(n), _attributes=attrs)

        for cg in genome.connections.values():
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid" if cg.enabled else "dotted"
            color = "green" if cg.weight > 0 else "red"
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={"style": style, "color": color, "penwidth": width})

        path = os.path.join(out_dir, f"best_genome_{generation}")
        graphviz.Source(dot.source, filename=path, format="jpeg").render(cleanup=True)

    def end_generation(self, config, population, species_set):
        self.visualize(self.out_dir, config, self.population.best_genome, self.population.generation)
        save_file(os.path.join(self.out_dir, f"best_genome_{self.population.generation}.pkl"), self.population.best_genome)
