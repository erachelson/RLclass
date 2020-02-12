from gym.envs.registration import register
import numpy as np
from core.mcts import MCTS, RootParentNode, Node
from env.cartpole import CartPole
import gym
import time


class MCTSAgent:
    def __init__(self, env_creator, config):
        self.env = env_creator()
        self.env_creator = env_creator
        self.config = config

    def play_episode(self):
        # reset env
        obs = self.env.reset()
        env_state = self.env.get_state()

        done = False
        t = 0
        total_reward = 0.0

        mcts = MCTS(self.config)

        root_node = Node(
            state=env_state,
            done=False,
            obs=obs,
            reward=0,
            action=None,
            parent=RootParentNode(env=self.env_creator()),
            mcts=mcts,
            depth=0
        )

        compute_action_times = []
        while not done:
            t += 1
            # compute action choice
            t0 = time.time()
            tree_policy, action, _, root_node = mcts.compute_action(root_node)
            root_node.parent = RootParentNode(env=self.env_creator())
            compute_action_times.append(time.time() - t0)

            # take action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

        avg_time = np.mean(compute_action_times)
        return t, total_reward, avg_time


if __name__ == "__main__":
    # instanciate env creator
    env_creator = lambda: CartPole()

    mcts_config = {
        "num_simulations": 10,
        "gamma": 0.997,
        "temperature": 1.0,
        "c1_coefficient": 1.25,
        "c2_coefficient": 19652
    }

    agent = MCTSAgent(env_creator, mcts_config)

    # play episodes
    for _ in range(10):
        t, total_reward, avg_time = agent.play_episode()
        print('Done, num timesteps: {}, reward: {}, averaged compute action time: {}'.format(t, total_reward, avg_time))
