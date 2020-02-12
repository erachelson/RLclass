import collections
from collections import OrderedDict
import math
import numpy as np


class Node:
    def __init__(self, action, reward, obs, state, mcts, depth, done, parent=None):

        self.env = parent.env
        self.action = action  # Action used to go to this state
        self.done = done

        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.depth = depth

        self.action_space_size = self.env.action_space.n
        self.child_q_value = np.zeros([self.action_space_size], dtype=np.float32)  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros([self.action_space_size], dtype=np.float32)  # N

        self.reward = reward
        self.obs = obs
        self.state = state

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @property
    def v_value(self):
        # get q_values only for actions that have been tried
        q_values = np.extract(self.child_number_visits > 0, self.child_q_value)
        assert len(q_values) > 0, 'error qvalues are empty'
        return np.mean(q_values)

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def q_value(self):
        return self.parent.child_q_value[self.action]

    @q_value.setter
    def q_value(self, value):
        self.parent.child_q_value[self.action] = value
        self.mcts.update_q_value_stats(value)

    def child_Q(self):
        return self.mcts.normalize_q_value(self.child_q_value)

    def child_U(self):
        c1, c2 = self.mcts.params['c1_coefficient'], self.mcts.params['c2_coefficient']
        utility = math.sqrt(self.number_visits) * self.child_priors / (1 + self.child_number_visits)
        utility *= (c1 + math.log((self.number_visits + c2 + 1) / c2))
        return utility

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.child_U()
        return np.argmax(child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            # call dynamics function g to obtain new state and reward
            self.env.set_state(self.state)
            obs, reward, done, info = self.env.step(action)
            next_state = self.env.get_state()

            self.children[action] = Node(
                obs=obs,
                done=done,
                state=next_state,
                action=action,
                depth=self.depth+1,
                parent=self,
                reward=reward,
                mcts=self.mcts,
            )
        return self.children[action]

    def backup(self, value):
        # update leaf node
        discount_return = value
        self.q_value = (self.number_visits * self.q_value + discount_return) / (1 + self.number_visits)
        self.number_visits += 1
        # update all other nodes up to root node
        current = self.parent
        while current.parent is not None:
            discount_return = current.reward + self.mcts.params['gamma'] * discount_return
            current.q_value = (current.number_visits * current.q_value + discount_return) / (1 + current.number_visits)
            current.number_visits += 1
            current = current.parent


class RootParentNode(object):
    def __init__(self, env):
        self.parent = None
        self.child_q_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.depth = 0
        self.env = env


class MCTS:
    def __init__(self, mcts_param):
        self.params = mcts_param
        self.max_q_value = -math.inf
        self.min_q_value = math.inf

    def update_q_value_stats(self, q_value):
        self.max_q_value = max(self.max_q_value, q_value)
        self.min_q_value = min(self.min_q_value, q_value)

    def normalize_q_value(self, q_value):
        if self.max_q_value > self.min_q_value:
            return (q_value - self.min_q_value) / (self.max_q_value - self.min_q_value)
        else:
            return q_value

    def compute_priors_and_value(self, node):
        env = node.env
        env.set_state(node.state)

        value = 0.0
        done = False
        t = 0
        while not done:
            _, reward, done, _ = env.step(env.action_space.sample())
            value += reward * (self.params['gamma']**t)
            t += 1

        priors = np.ones((1, node.action_space_size)) / node.action_space_size
        return priors, value

    def compute_action(self, node):
        for _ in range(self.params['num_simulations']):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.compute_priors_and_value(leaf)
                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(tree_policy)  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, 1/self.params['temperature'])
        tree_policy = tree_policy / np.sum(tree_policy)

        # Choose action according to tree policy
        action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.v_value, node.children[action]
