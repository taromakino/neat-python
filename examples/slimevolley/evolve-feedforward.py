"""
Single-pole balancing experiment using a feed-forward neural network.
"""
import argparse

import jax
import jax.numpy as jnp
import multiprocessing
import neat
import numpy as np
import os
from argparse import ArgumentParser
from feed_forward import BatchFeedForwardNetwork
from functools import partial
from slimevolley import SlimeVolley
from metrics_reporter import MetricsReporter
from visualize_reporter import VisualizeReporter
from identity_output_reporter import IdentityOutputReporter


def eval_genome(
        genome: neat.DefaultGenome,
        config: neat.Config,
        batch_size: int
) -> float:
    net = BatchFeedForwardNetwork.create(genome, config)

    task = SlimeVolley()
    task_reset_fn = jax.jit(task.reset)
    task_step_fn = jax.jit(task.step)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    task_states = task_reset_fn(keys)
    rewards = jnp.zeros(batch_size)
    done = jnp.zeros((batch_size), dtype=bool)

    while not done.all():
        actions = net.activate(np.array(task_states.obs))
        task_states, step_rewards, done = task_step_fn(task_states, jnp.array(actions))
        rewards = rewards + step_rewards

    return rewards.mean().item()


def save_gif(out_dir, best_genome, config):
    net = BatchFeedForwardNetwork.create(best_genome, config)

    task = SlimeVolley(test=True)
    task_reset_fn = jax.jit(task.reset)
    task_step_fn = jax.jit(task.step)

    keys = jax.random.PRNGKey(0)[None, :]
    task_state = task_reset_fn(keys)
    done = jnp.zeros((1), dtype=bool)

    screens = []
    while not done:
        actions = net.activate(np.array(task_state.obs))
        task_state, step_rewards, done = task_step_fn(task_state, jnp.array(actions))
        screens.append(SlimeVolley.render(task_state))

    os.makedirs(out_dir, exist_ok=True)
    gif_file = os.path.join(out_dir, "slimevolley.gif")
    screens[0].save(gif_file, save_all=True, append_images=screens[1:1000], duration=20, loop=0)


def set_identity_output_activations(population) -> None:
    """
    Fix the output activations to the identity function.
    """
    for genome in population.population.values():
        for node_key in range(population.config.genome_config.num_outputs):
            genome.nodes[node_key].activation = "identity"


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    set_identity_output_activations(population)

    stats = neat.StatisticsReporter()
    metrics = MetricsReporter(args.out_dir, population)
    visualize = VisualizeReporter(args.out_dir, population)
    identity_output_activations = IdentityOutputReporter(partial(
        set_identity_output_activations,
        population=population
    ))
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(metrics)
    population.add_reporter(visualize)
    population.add_reporter(identity_output_activations)

    # Iterate through the generations, running each genome on a separate process
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), partial(eval_genome, batch_size=args.batch_size))
    winner = population.run(pe.evaluate, n=args.num_generations)
    print(winner)

    save_gif(args.out_dir, population.best_genome, config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    main(parser.parse_args())