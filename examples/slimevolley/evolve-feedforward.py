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
from feed_forward import FeedForwardNetwork
from functools import partial
from slimevolley import SlimeVolley


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


def save_gif(out_dir, best_genome, config):
    net = FeedForwardNetwork.create(best_genome, config)

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
    screens[0].save(gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0)


def main(args):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), partial(eval_genome, batch_size=args.batch_size))
    winner = pop.run(pe.evaluate, n=args.num_generations)

    print(winner)

    save_gif(args.out_dir, pop.best_genome, config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    main(parser.parse_args())
