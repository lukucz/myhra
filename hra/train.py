import os
import yaml
import click

import sys
sys.path.append('..')

from environment.my_fruit_collection import FruitCollectionMini
from hra.ai import AI

import tensorflow as tf
import numpy as np

# Tensorflow session config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def set_params(params, mode):
    if mode == 'hra':
        params['gamma'] = .99
        params['learning_rate'] = .001
    elif mode == 'hra+1':
        params['gamma'] = .99
        params['learning_rate'] = .001
        params['remove_features'] = True
    elif mode == 'hra+2':
        params['gamma'] = .99
        params['learning_rate'] = .001
        params['remove_features'] = True
        params['no_terminal_state'] = True

    return params


def agent(params):
    np.random.seed(seed=params['random_seed'])
    rng = np.random.RandomState(params['random_seed'])

    env = FruitCollectionMini(state_mode='mini', game_length=300)

    ai = AI(env,
            minibatch_size=params['minibatch_size'], gamma=params['gamma'], learning_rate=params['learning_rate'],
            epsilon=params['epsilon'], final_epsilon=params['final_epsilon'],
            epsilon_decaying_frames=params['epsilon_decaying_frames'], update_freq=params['update_freq'],
            max_iterations=params['max_iterations'], replay_min_size=params['replay_min_size'],
            replay_max_size=params['replay_max_size'], num_units=params['num_units'],
            remove_features=params['remove_features'], no_terminal_state=params['no_terminal_state'])


@click.command()
@click.option('--mode', default='hra', help='Which method to run: hra, hra+1, hra+2')
def main(mode):
    # config.yml 読み込み
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = os.path.join(dir_path, 'config.yaml')
    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    # mode 指定
    valid_modes = ['hra', 'hra+1', 'hra+2']
    assert mode in valid_modes
    set_params(params, mode)

    # 学習開始
    agent(params)


if __name__ == '__main__':
    main()
