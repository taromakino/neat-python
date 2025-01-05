"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import jax
import jax.numpy as jnp
import multiprocessing
import neat
import numpy as np
import os
from argparse import ArgumentParser
from evojax.task.slimevolley import SlimeVolley
from feed_forward import FeedForwardNetwork
from functools import partial


def eval_genome(genome, config, batch_size):
    net = FeedForwardNetwork.create(genome, config)

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


def eval_genomes(genomes, config, batch_size):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, batch_size)


def main(args):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # winner = pop.run(partial(eval_genomes, batch_size=args.batch_size), 300)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), partial(eval_genome, batch_size=args.batch_size))
    winner = pop.run(pe.evaluate)

    print(winner)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    main(parser.parse_args())
