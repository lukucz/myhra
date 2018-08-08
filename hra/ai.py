import tensorflow as tf
import numpy as np
import random
from copy import deepcopy
from collections import deque
from model import DeepQNetwork as dqn
from time import sleep


class AI(object):
    def __init__(self, env, minibatch_size=32, gamma=0.99, learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, epsilon_decaying_frames=1, update_freq=100, max_iterations=1e8, replay_min_size=1000, replay_max_size=10000, num_units=250, remove_features=False, no_terminal_state=False):
        # environment
        self.env = env
        self.state_shape = env.state_shape_net
        self.num_actions = env.nb_actions
        self.possible_actions = env.legal_actions
        self.possible_fruits = env.possible_fruits
        self.num_rewards = len(self.possible_fruits)
        self.remove_features = remove_features
        self.no_terminal_state = no_terminal_state
        if remove_features:
            self.state_shape = self.state_shape[: -1] + \
                [self.state_shape[-1] - self.num_rewards + 1]

        # learning
        self.num_units = num_units
        self.minibatch_size = minibatch_size
        self.batch_state_shape = tuple([minibatch_size] + self.state_shape)
        self.action_state_shape = tuple([1] + self.state_shape)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decaying_frames = epsilon_decaying_frames
        self.update_freq = update_freq
        self.max_iterations = max_iterations

        # replay memory
        self.replay_min_size = replay_min_size
        self.replay_max_size = replay_max_size
        self.replay_memory = deque(maxlen=self.replay_max_size)

        # networks
        self.network_list = []
        self.train_index = 0
        self.target_index = 1
        self.target_q_index = 2
        self.action_q_index = 3
        self.loss_index = 4
        self.optimizer_index = 5

        # Start Training
        self.init()
        self.start_training()

    def init(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 学習用の入力
            self.tf_train_input = tf.placeholder(tf.float32, shape=self.batch_state_shape)
            self.tf_train_target = tf.placeholder(
                tf.float32, shape=(self.minibatch_size, self.num_actions))
            self.tf_filter_input = tf.placeholder(
                tf.float32, shape=(self.minibatch_size, self.num_actions))
            self.tf_target_input = tf.placeholder(tf.float32, shape=self.batch_state_shape)
            self.tf_action_selection_input = tf.placeholder(
                tf.float32, shape=self.action_state_shape)

            # train network : 学習するネットワーク
            # target network : 学習するときに参照するネットワーク、定期更新
            for _ in range(self.num_rewards):
                train_network = dqn(self.state_shape, int(self.num_units / self.num_rewards),
                                    self.num_actions, 'train')
                target_network = dqn(self.state_shape, int(self.num_units / self.num_rewards),
                                     self.num_actions, 'target')
                target_q_values = target_network.q_values(self.tf_target_input)
                action_q_values = train_network.q_values(self.tf_action_selection_input)
                loss = train_network.clipped_loss(
                    self.tf_train_input, self.tf_train_target, self.tf_filter_input)
                optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate, decay=.95, epsilon=1e-7).minimize(loss)
                self.network_list.append([train_network, target_network,
                                          target_q_values, action_q_values,
                                          loss, optimizer])

    def start_training(self):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            self.update_target_network(session)

            games = 0           # ゲームの回数
            steps = 0           # 移動回数
            total_reward = 0    # 1ゲームにおける報酬の合計
            iterations = 0      # 学習回数

            # training
            while iterations < self.max_iterations:
                # game init
                self.env.reset()
                total_reward = 0
                steps = 0
                prev_state = self.env.get_state()
                games += 1

                while not self.env.game_over:
                    # select action
                    action = self.select_action(prev_state, self.possible_actions, games)
                    steps += 1
                    state, reward, terminal, position = self.env.step(action)

                    # save experience
                    if not terminal:
                        self.save_replay_memory((prev_state, action, reward, state, terminal))
                    else:
                        if not self.no_terminal_state:
                            self.save_replay_memory((prev_state, action, reward, state, terminal))

                    # training (ExperienceReplay)
                    if self.has_enough_memory():
                        if iterations % self.update_freq == 0:
                            self.update_target_network(session)
                        memories = self.sample_replay_memory()
                        self.train_with(session, memories)
                        iterations += 1

                    prev_state = state
                    total_reward += np.sum(reward)

                # log
                print('game: ', games, 'steps: ', steps, 'iterations: ', iterations,
                      'total_reward: ', total_reward)

    def _remove_features(self, s, i):
        return np.append(s[:-self.num_rewards], s[self.state_shape[-1] - 1 + i])

    def update_target_network(self, session):
        for i in range(self.num_rewards):
            self.network_list[i][self.train_index].copy_network_to(
                self.network_list[i][self.target_index], session)

    def is_greedy(self, epsilon):
        return epsilon < np.random.rand()

    def select_greedy_action_from(self, state, available_actions):
        q = []
        for i in range(self.num_rewards):
            local_s = state
            if self.remove_features:
                local_s = self._remove_features(local_s, i)
            q_values = self.network_list[i][self.action_q_index].eval(
                feed_dict={self.tf_action_selection_input: [local_s]})
            q.append(q_values[0])

        mean = np.average(q, axis=0)
        index = np.argmax(mean)

        if index in available_actions:
            return index
        else:
            return np.random.choice(available_actions)

    def select_random_action_from(self, available_actions):
        return np.random.choice(available_actions)

    def calculate_epsilon(self, frame_num):
        if self.epsilon_decaying_frames <= frame_num:
            return self.final_epsilon
        else:
            return 1.0 - ((1.0 - self.final_epsilon) / self.epsilon_decaying_frames) * frame_num

    def select_action(self, state, actions, frame_num):
        epsilon = self.calculate_epsilon(frame_num)
        if self.is_greedy(epsilon):
            action = self.select_greedy_action_from(state, actions)
        else:
            action = self.select_random_action_from(actions)
        return action

    def has_enough_memory(self):
        return self.replay_min_size <= len(self.replay_memory)

    def sample_replay_memory(self):
        return random.sample(self.replay_memory, self.minibatch_size)

    def save_replay_memory(self, memory):
        if memory is None:
            return
        self.replay_memory.append(memory)

    def train_with(self, session, memories):
        # memory = (state, action, reward, next_state, terminal)
        action_batch = [memory[1] for memory in memories]
        reward_batch = [memory[2] for memory in memories]
        if not self.remove_features:
            state_batch = [memory[0] for memory in memories]
            next_state_batch = [memory[3] for memory in memories]

        for i in range(self.num_rewards):
            if self.remove_features:
                state_batch = [self._remove_features(memory[0], i) for memory in memories]
                next_state_batch = [self._remove_features(memory[3], i) for memory in memories]

            target_qs = self.network_list[i][self.target_q_index].eval(
                feed_dict={self.tf_target_input: next_state_batch})
            target = np.zeros(shape=(self.minibatch_size, self.num_actions), dtype=np.float32)
            q_value_filter = np.zeros(
                shape=(self.minibatch_size, self.num_actions), dtype=np.float32)

            for j in range(self.minibatch_size):
                end_state = memories[j][4]
                action_index = action_batch[j]
                reward = reward_batch[j][i]
                target[j][action_index] = reward if end_state else reward + \
                    self.gamma * np.max(target_qs[j])
                q_value_filter[j][action_index] = 1.0

            session.run(self.network_list[i][self.optimizer_index], feed_dict={
                        self.tf_train_input: state_batch,
                        self.tf_train_target: target,
                        self.tf_filter_input: q_value_filter})
