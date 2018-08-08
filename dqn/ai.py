import tensorflow as tf
import numpy as np
import random
from copy import deepcopy
from collections import deque
from model import DeepQNetwork as dqn


class AI(object):
    def __init__(self, env, minibatch_size=32, gamma=0.99, learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, epsilon_decaying_frames=1, update_freq=100, max_iterations=1e8, replay_min_size=1000, replay_max_size=10000, num_units=250):
        # environment
        self.env = env
        self.state_shape = env.state_shape_net
        self.num_actions = env.nb_actions
        self.possible_actions = env.legal_actions
        self.possible_fruits = env.possible_fruits
        self.num_rewards = len(self.possible_fruits)

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

        # Start Training
        self.init()
        self.start_training()

    def init(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_input = tf.placeholder(tf.float32, shape=self.batch_state_shape)
            self.tf_train_target = tf.placeholder(
                tf.float32, shape=(self.minibatch_size, self.num_actions))
            self.tf_filter_input = tf.placeholder(
                tf.float32, shape=(self.minibatch_size, self.num_actions))
            self.tf_target_input = tf.placeholder(tf.float32, shape=self.batch_state_shape)
            self.tf_action_selection_input = tf.placeholder(
                tf.float32, shape=self.action_state_shape)

            self.train_network = dqn(self.state_shape, self.num_units,
                                     self.num_rewards, self.num_actions, 'train')
            self.target_network = dqn(self.state_shape, self.num_units,
                                      self.num_rewards, self.num_actions, 'target')
            self.target_q_values = self.target_network.q_values(self.tf_target_input)
            self.action_q_values = self.train_network.q_values(self.tf_action_selection_input)

            self.loss = self.train_network.clipped_loss(
                self.tf_train_input, self.tf_train_target, self.tf_filter_input)
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate, decay=.95, epsilon=1e-7).minimize(self.loss)

    def start_training(self):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            self.update_target_network(session)

            games = 0
            steps = 0
            total_reward = 0
            iterations = 0

            while iterations < self.max_iterations:
                # training
                self.env.reset()
                total_reward = 0
                steps = 0
                prev_state = self.env.get_state()
                games += 1
                while not self.env.game_over:
                    action = self.select_action(prev_state, self.possible_actions, games)
                    steps += 1
                    state, reward, terminal, position = self.env.step(action)
                    sum_reward = np.sum(reward)

                    self.save_replay_memory((prev_state, action, sum_reward, state, terminal))

                    if self.has_enough_memory():
                        if iterations % self.update_freq == 0:
                            self.update_target_network(session)
                        memories = self.sample_replay_memory()
                        self.train_with(session, memories)
                        iterations += 1

                    prev_state = state
                    total_reward += sum_reward

                print('game: ', games, 'steps: ', steps, 'iterations: ', iterations,
                      'total_reward: ', total_reward)

    def update_target_network(self, session):
        self.train_network.copy_network_to(self.target_network, session)

    def is_greedy(self, epsilon):
        return epsilon < np.random.rand()

    def select_greedy_action_from(self, state, available_actions):
        q_values = self.action_q_values.eval(
            feed_dict={self.tf_action_selection_input: [state]})
        index = np.argmax(q_values)
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
        state_batch = [memory[0] for memory in memories]
        action_batch = [memory[1] for memory in memories]
        reward_batch = [memory[2] for memory in memories]
        next_state_batch = [memory[3] for memory in memories]

        target_qs = self.target_q_values.eval(feed_dict={self.tf_target_input: next_state_batch})
        target = np.zeros(shape=(self.minibatch_size, self.num_actions), dtype=np.float32)
        q_value_filter = np.zeros(shape=(self.minibatch_size, self.num_actions), dtype=np.float32)

        for i in range(self.minibatch_size):
            end_state = memories[i][4]
            action_index = action_batch[i]
            reward = reward_batch[i]
            target[i][action_index] = reward if end_state else reward + \
                self.gamma * np.max(target_qs[i])
            q_value_filter[i][action_index] = 1.0

        session.run(self.optimizer, feed_dict={self.tf_train_input: state_batch,
                                               self.tf_train_target: target,
                                               self.tf_filter_input: q_value_filter})
