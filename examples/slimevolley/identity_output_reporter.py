from neat.reporting import BaseReporter


class IdentityOutputReporter(BaseReporter):
    def __init__(self, set_identity_output_activations_fn):
        self.set_identity_output_activations_fn = set_identity_output_activations_fn

    def write(self, path, text):
        with open(path, "a+") as f:
            f.write(text + "\n")

    def end_generation(self, config, population, species_set):
        self.set_identity_output_activations_fn()