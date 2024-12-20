"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle

import jax
import jax.numpy as jnp
import neat
import visualize

from evojax.task.slimevolley import SlimeVolley


NUM_REPEATS = 16


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = SlimeVolley()

    # Create batched random keys
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, NUM_REPEATS)

    # Reset multiple environments at once
    state = env.reset(keys)
    terminated = jnp.zeros(NUM_REPEATS, dtype=bool)
    total_rewards = jnp.zeros(NUM_REPEATS)

    while not jnp.all(terminated):
        # Get observations for all non-terminated environments
        obs_batch = state.obs.reshape(NUM_REPEATS, -1)

        # Process each observation through the network
        actions = []
        for obs in obs_batch:
            action = net.activate(obs.tolist())
            actions.append(action)
        actions = jnp.array(actions)

        # Step all environments
        state, reward, new_terminated = env.step(state, actions)

        # Update total rewards for non-terminated environments
        total_rewards = jnp.where(terminated, total_rewards, total_rewards + reward)
        terminated = jnp.logical_or(terminated, new_terminated)

    # Return the mean reward across all batches
    return total_rewards.mean().item()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    run()
