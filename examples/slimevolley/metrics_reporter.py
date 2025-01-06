import os
from neat.reporting import BaseReporter


class MetricsReporter(BaseReporter):
    def __init__(self, out_dir, population):
        self.out_dir = out_dir
        self.population = population
        self.metrics_path = os.path.join(self.out_dir, "metrics.csv")
        self.write(self.metrics_path, "generation,best_fitness")

    def write(self, path, text):
        with open(path, "a+") as f:
            f.write(text + "\n")

    def end_generation(self, config, population, species_set):
        self.write(self.metrics_path, f"{self.population.generation},{self.population.best_genome.fitness}")