import os
from copy import deepcopy
import numpy as np


# RGB colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
WALL = (80, 80, 80)


class FruitCollection(object):
    def __init__(self, game_length=300, lives=1e6, state_mode='pixel', is_fruit=True, is_ghost=True, rng=None):
        self.game_length = game_length
        self.lives = lives
        self.is_fruit = is_fruit
        self.is_ghost = is_ghost
        self.legal_actions = [0, 1, 2, 3]
        self.action_meanings = ['up', 'down', 'left', 'right']
        self.reward_scheme = {'ghost': -10.0, 'fruit': +1.0, 'step': 0.0, 'wall': 0.0}
        self.nb_actions = len(self.legal_actions)
        if rng is None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng
        self.player_pos_x = None
        self.player_pos_y = None
        self.agent_init_pos = None
        self.pass_wall_rows = None
        self.init_lives = deepcopy(self.lives)
        self.step_reward = 0.0
        self.possible_fruits = []

        # how the returned state look like ('pixel' or '1hot' or 'multi-head')
        self.state_mode = state_mode

        self.nb_fruits = None
        self.scr_w = None
        self.scr_h = None
        self.rendering_scale = None
        self.walls = None
        self.fruits = None
        self.ghosts = None
        self.init_with_mode()
        self.nb_non_wall = self.scr_w * self.scr_h - len(self.walls)
        self.init_ghosts = deepcopy(self.ghosts)

        self.targets = None  # fruits + ghosts
        self.active_targets = None  # boolean list
        self.active_fruits = None
        self.nb_targets = None
        self.init_targets = None
        self.nb_ghosts = None
        self.soc_state_shape = None
        self.state_shape = None
        self.state_shape_net = None
        self.state = None
        self.step_id = 0
        self.game_over = False
        self.mini_target = []  # only is used for mini
        self.reset()

    def init_with_mode(self):
        raise NotImplementedError

    def reset(self):
        self.game_over = False
        self.step_id = 0
        self._reset_targets()
        self.nb_ghosts = len(self.ghosts)
        self.targets = deepcopy(self.fruits) + deepcopy(self.ghosts)
        self.nb_targets = len(self.targets)
        self.active_targets = self.active_fruits + [True] * len(self.ghosts)
        self.lives = deepcopy(self.init_lives)
        self.soc_state_shape = [self.scr_w, self.scr_h, self.scr_w + 1, self.scr_h + 1]
        if self.state_mode == 'pixel':
            self.state_shape = [4, self.scr_w, self.scr_h]
            self.state_shape_net = [self.scr_w, self.scr_h, 4]
        elif self.state_mode == 'mini':
            self.state_shape = [100 + len(self.possible_fruits)]
            self.state_shape_net = [100 + len(self.possible_fruits)]

    def _reset_targets(self):
        raise NotImplementedError

    def _move_player(self, action):
        assert action in self.legal_actions, 'Illegal action.'
        hit_wall = False
        if action == 3:  # right
            passed_wall = False
            if self.pass_wall_rows is not None:
                for wall_row in self.pass_wall_rows:
                    if [self.player_pos_x, self.player_pos_y] == [self.scr_w - 1, wall_row]:
                        self.player_pos_x = 0
                        passed_wall = True
                        break
            if not passed_wall:
                if [self.player_pos_x + 1, self.player_pos_y] not in self.walls and self.player_pos_x < self.scr_w - 1:
                    self.player_pos_x += 1
                else:
                    hit_wall = True
        elif action == 2:  # left
            passed_wall = False
            if self.pass_wall_rows is not None:
                for wall_row in self.pass_wall_rows:
                    if [self.player_pos_x, self.player_pos_y] == [0, wall_row]:
                        self.player_pos_x = self.scr_w - 1
                        passed_wall = True
                        break
            if not passed_wall:
                if [self.player_pos_x - 1, self.player_pos_y] not in self.walls and self.player_pos_x > 0:
                    self.player_pos_x -= 1
                else:
                    hit_wall = True
        elif action == 1:  # down
            if [self.player_pos_x, self.player_pos_y + 1] not in self.walls and self.player_pos_y < self.scr_h - 1:
                self.player_pos_y += 1
            else:
                hit_wall = True
        elif action == 0:  # up
            if [self.player_pos_x, self.player_pos_y - 1] not in self.walls and self.player_pos_y > 0:
                self.player_pos_y -= 1
            else:
                hit_wall = True
        return hit_wall

    def _check_fruit(self):
        if not self.is_fruit:
            return None
        caught_target = None
        caught_target_idx = None
        target_count = -1
        for k, target in enumerate(self.targets):
            target_count += 1
            if target['reward'] < 0:  # not fruit
                continue
            if target['location'] == [self.player_pos_x, self.player_pos_y] and target['active'] is True:
                caught_target = deepcopy([self.player_pos_y, self.player_pos_x])
                caught_target_idx = k
                target['active'] = False
                target['location'] = [self.scr_w, self.scr_h]  # null value
                break
        check = []
        for target in self.targets:
            if target['reward'] > 0:
                check.append(target['active'])
        if True not in check:
            self.game_over = True
        return caught_target, caught_target_idx

    def get_state(self):
        if self.state_mode == 'pixel':
            return self.get_state_pixel()
        elif self.state_mode == 'mini':
            return self.get_mini_state()
        else:
            raise ValueError('State-mode is not known.')

    def get_mini_state(self):
        state = np.zeros((self.scr_w * self.scr_h + len(self.possible_fruits)), dtype=np.int8)
        state[self.player_pos_y * self.scr_h + self.player_pos_x] = 1
        for target in self.targets:
            if target['active'] and target['reward'] > 0:
                offset = self.possible_fruits.index([target['location'][1], target['location'][0]])
                index = (self.scr_w * self.scr_h) + offset
                state[index] = 1
        return state

    def get_state_pixel(self):
        state = np.zeros((self.state_shape[1], self.state_shape[2],
                          self.state_shape[0]), dtype=np.int8)
        # walls, fruits, player, ghost
        player_pos = [self.player_pos_x, self.player_pos_y]
        fruits = []
        ghosts = []
        for target in self.targets:
            if target['active'] is True:
                if target['reward'] > 0:
                    fruits.append(target['location'])
                elif target['reward'] < 0:
                    ghosts.append(target['location'])
        for loc in fruits:
            if loc in ghosts and self.is_ghost:
                # state[tuple(loc)] = self.code['fruit+ghost']
                state[tuple(loc)][1] = 1
                state[tuple(loc)][3] = 1
                ghosts.remove(loc)
            else:
                state[tuple(loc)][1] = 1
                # state[tuple(loc)] = self.code['fruit']
        if player_pos in ghosts and self.is_ghost:
            state[tuple(player_pos)][2] = 1
            state[tuple(player_pos)][3] = 1
            ghosts.remove(player_pos)
        else:
            state[tuple(player_pos)][2] = 1
        if self.is_ghost:
            for loc in ghosts:
                state[tuple(loc)][3] = 1
                # state[tuple(loc)] = self.code['ghost']
        for loc in self.walls:
            state[tuple(loc)][0] = 1
            # state[tuple(loc)] = self.code['wall']
        return state

    def step(self, action):
        # actions: [0, 1, 2, 3] == [up, down, left, right]
        player_pos = [self.player_pos_x, self.player_pos_y]
        if self.game_over:
            raise ValueError('Environment has already been terminated.')
        if self.step_id >= self.game_length - 1:
            self.game_over = True
            head_reward = [0 for i in range(len(self.possible_fruits))]
            return self.get_state(), head_reward, self.game_over, player_pos
        last_player_position = deepcopy([self.player_pos_x, self.player_pos_y])
        hit_wall = self._move_player(action)
        if hit_wall:
            wall_reward = self.reward_scheme['wall']
        else:
            wall_reward = 0.0

        caught_fruit, caught_fruit_idx = self._check_fruit()

        head_reward = [0 for i in range(len(self.possible_fruits))]

        if caught_fruit is not None:
            head_reward[self.possible_fruits.index(caught_fruit)] = 1

        self.step_id += 1

        return self.get_state(), head_reward, self.game_over, player_pos


class FruitCollectionMini(FruitCollection):
    def init_with_mode(self):
        self.is_ghost = False
        self.is_fruit = True
        self.nb_fruits = 5
        self.possible_fruits = [[0, 0], [0, 9], [1, 2], [3, 6],
                                [4, 4], [5, 7], [6, 2], [7, 7], [8, 8], [9, 0]]
        self.scr_w = 10
        self.scr_h = 10
        self.rendering_scale = 50
        self.walls = []
        if self.is_ghost:
            self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [0, 1],
                            'active': True}]
        else:
            self.ghosts = []

    def _reset_targets(self):
        while True:
            self.player_pos_x, self.player_pos_y = self.rng.randint(
                0, self.scr_w), self.rng.randint(0, self.scr_h)
            if [self.player_pos_x, self.player_pos_y] not in self.possible_fruits:
                break
        # Targets:  Format: [ {colour: c1, reward: r1, locations: list_l1, 'active': list_a1}, ... ]
        self.fruits = []
        self.active_fruits = []
        if self.is_fruit:
            for x in range(self.scr_w):
                for y in range(self.scr_h):
                    self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                        'location': [x, y], 'active': False})
                    self.active_fruits.append(False)
            fruits_idx = deepcopy(self.possible_fruits)
            self.rng.shuffle(fruits_idx)
            fruits_idx = fruits_idx[:self.nb_fruits]
            self.mini_target = [False] * len(self.possible_fruits)
            for f in fruits_idx:
                idx = f[1] * self.scr_w + f[0]
                self.fruits[idx]['active'] = True
                self.active_fruits[idx] = True
                self.mini_target[self.possible_fruits.index(f)] = True
