import os
import yaml
import click

import sys
sys.path.append('..')

from environment.my_fruit_collection import FruitCollectionMini
from dqn.ai import AI

import tensorflow as tf
import numpy as np

# Tensorflow session config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def agent(params):
    np.random.seed(seed=params['random_seed'])
    rng = np.random.RandomState(params['random_seed'])

    env = FruitCollectionMini(state_mode='mini', game_length=300)

    ai = AI(env,
            minibatch_size=params['minibatch_size'], gamma=params['gamma'], learning_rate=params['learning_rate'],
            epsilon=params['epsilon'], final_epsilon=params['final_epsilon'],
            epsilon_decaying_frames=params['epsilon_decaying_frames'], update_freq=params['update_freq'],
            max_iterations=params['max_iterations'], replay_min_size=params['replay_min_size'],
            replay_max_size=params['replay_max_size'], num_units=params['num_units'])


@click.command()
@click.option('--mode', default='dqn', help='Which method to run: dqn')
def main(mode):
    # config.yml 読み込み
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = os.path.join(dir_path, 'config.yaml')
    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    # 学習開始
    agent(params)


if __name__ == '__main__':
    main()
