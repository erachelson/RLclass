import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame

import copy
import numpy as np

class weightedDiscrete(spaces.Discrete):
    def sample(self):
        return np.random.choice(np.arange(self.n), p=(0.8,0.2))


class GymFlappy(gym.Env, EzPickle):
    def __init__(self, config=None):
        EzPickle.__init__(self)

        # Aid options
        self.pre_play = True
        self.force_calm = False
        self.positive_counts = 0

        self.display_screen = False
        if config:
            self.display_screen = config['display_screen']

        self.observation_space = spaces.Box(0, 1, shape=(8,), dtype=np.float32)
        self.action_space = weightedDiscrete(2) #spaces.Discrete(2)

        self.vel_max = 15
        self.vel_min = -15
        self.dist_max = 500
        self.dist_min = 0
        self.y_max = 500
        self.y_min = 0

        self.game = FlappyBird(graphics="fancy")
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=self.display_screen, rng=0)
        self.p.rng = self.game.rng
        self.game.player.rng = self.game.rng
        
        self.p.init()

        self.current_t = 0
        self.max_t = 1000

    def _get_obs(self):
        state = self.game.getGameState()
        obs = np.empty((8,))
        obs[0] = (state["player_y"] - self.y_min) / (self.y_max - self.y_min)
        obs[1] = (state["next_pipe_dist_to_player"] - self.dist_min) / (self.dist_max - self.dist_min)
        obs[2] = (state["next_pipe_top_y"] - self.y_min) / (self.y_max - self.y_min)
        obs[3] = (state["next_pipe_bottom_y"] - self.y_min) / (self.y_max - self.y_min)
        obs[4] = (state["next_next_pipe_dist_to_player"] - self.dist_min) / (self.dist_max - self.dist_min)
        obs[5] = (state["next_next_pipe_top_y"] - self.y_min) / (self.y_max - self.y_min)
        obs[6] = (state["next_next_pipe_bottom_y"] - self.y_min) / (self.y_max - self.y_min)
        obs[7] = (state["player_vel"] - self.vel_min) / (self.vel_max - self.vel_min)
        return obs

    def reset(self):
        self.current_t = 0
        self.p.reset_game()

        if self.pre_play: # Get rid of the first second of game
            ini_fc = self.force_calm
            self.force_calm = False
            for i in range(25):
                a = 0
                if i % 10 == 0:
                    a = 1
                self.step(np.array([a]))
            self.force_calm = ini_fc
        
        return self._get_obs()

    def step(self, action):
        self.current_t += 1
        reward = self.p.act(119 if action == 1 else 0)

        if self.force_calm: # ensures each action is followed by no action
            for i in range(1):
                r = self.p.act(0)
            reward += r
        
        done = self.current_t >= self.max_t or self.p.game_over()

        done = done or self._double_check_done()

        info = {}
        return self._get_obs(), reward, done, info

    def __getstate__(self):

        dc = lambda x : copy.deepcopy(x)

        # get all game attributes
        _game_state = self.game.__dict__
        _player_state = self.game.player.__dict__
        _pipe_state = self.game.pipe_group.__dict__
        pipe_sprites = self.game.pipe_group.spritedict
        pipe_xs = []
        pipe_ys = []
        pipe_rects = []
        for _, sprite in enumerate(pipe_sprites):
            pipe_xs.append(dc(sprite.x))
            pipe_ys.append(dc(sprite.gap_start))
            pipe_rects.append(dc(pipe_sprites[sprite]))
        lives = dc(self.game.lives)
        score = dc(self.game.getScore())
        pscore = dc(self.p.previous_score)

        # remove images (heavy and require additional serialization):
        __game_state = {}
        __player_state = {}
        for attr in _game_state:
            if attr in ['screen', 'images', 'clock','player', 'backdrop', "pipe_group"]:
                pass
            else:
                __game_state[attr] = _game_state[attr]
        for attr in _player_state:
            if attr in ['image', 'image_assets']:
                pass
            else:
                __player_state[attr] = _player_state[attr]

        # accomodate multiple envs in parallel
        game_state = dc(__game_state)
        player_state = dc(__player_state)
        pipe_state = _pipe_state

        # this is a non-PLE parameter that needs to be reset too
        envtime = dc(self.current_t)
        rng_state = self.game.rng.get_state()
        

        stategroup = (game_state, player_state, pipe_state, (pipe_xs, pipe_rects, pipe_ys), lives, envtime, rng_state, score, pscore)
        return stategroup

    def __setstate__(self, stategroup):
        '''
        Stategroup required (ugly yet somewhat functional):

        0   game_state dictionary (game.__dict__)
        1   player_state dictionary (game.player.__dict__)
        2   pipe_state idctionary (game.pipe_group.__dict__)
        3   x positions of pipes in game (list)
        4   lives (game.lives, used in game.game_over())
        5   current time (self.current_t)
        6   rng state
        '''
        # use update to preserve images we didn't save
        self.game.__dict__.update(stategroup[0])
        self.game.player.__dict__.update(stategroup[1])
        #self.game.pipe_group.__dict__.update(stategroup[2]) # was introducing reference crossing
        pipe_sprites = self.game.pipe_group.spritedict
        for i, sprite in enumerate(pipe_sprites):
            sprite.x = stategroup[3][0][i]
            pipe_sprites[sprite] = stategroup[3][1][i]
            sprite.gap_start = stategroup[3][2][i]

        self.game.lives = stategroup[4]

        # prevent Gym env to return false dones
        self.current_t = stategroup[5]
        self.game.rng.set_state(stategroup[6])

        # fix stupid reward
        self.game.score = stategroup[7]
        self.p.previous_score = stategroup[8]
        return self._get_obs()

    def get_state(self):
        return self.__getstate__()

    def set_state(self, state):
        return self.__setstate__(state)
    
    def reset_counts(self):
        self.positive_counts = 0
    
    def _double_check_done(self):
        '''
        Manually inspects game to detect collisions
        Worthy of suicide but necessary...
        '''

        # Check pipe collisions
        for p in self.game.pipe_group:
            hit = pygame.sprite.spritecollide(self.game.player, self.game.pipe_group, False)
            is_in_pipe = (p.x - p.width/2 - 20) <= self.game.player.pos_x < (p.x + p.width/2)
            for h in hit:  # do check to see if its within the gap.
                top_pipe_check = (
                    (self.game.player.pos_y - self.game.player.height/2 + 12) <= h.gap_start) and is_in_pipe
                bot_pipe_check = (
                    (self.game.player.pos_y +
                    self.game.player.height) > h.gap_start +
                    self.game.pipe_gap) and is_in_pipe
                boom = bot_pipe_check or top_pipe_check
                if boom:
                    return True
        
        # floor limit
        if self.game.player.pos_y >= 0.79 * self.game.height - self.game.player.height:
            return True
            
        # went above the screen
        if self.game.player.pos_y <= 0: return True

        return False