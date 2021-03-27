import gym
import random
import os
import numpy as np


# Defines a Q table given a discrete action space and continuous observation space
# This is well suited for video games that use a discrete set of buttons (not including press sensitivity  etc.)
class QTable:
    # Given a float n-dimensional Box, we will split the observation space disc_obsv_dims x action_space.n table
    def __init__(self, env, disc_obsv_dims, pretrained_table=None):
        # make sure the given observation space input matches the dimensions of the observation space of the environment
        if len(env.observation_space.high) != len(disc_obsv_dims):
            raise TypeError('\nGiven dimension of discrete observation space mismatch\n' +
                            'Given: ' + str(len(disc_obsv_dims)) + '\n' +
                            'Environment: ' + str(len(env.observation_space.high)))

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # figure out how many entries we need per dimension of the observation space
        self.dimensions = np.array(disc_obsv_dims + [self.action_space.n])

        # generate the q-table given that the action space is discrete given
        if pretrained_table is None:
            self.q_table = np.random.uniform(low=-2, high=0, size=self.dimensions)
        # load a pre-trained table for further testing
        else:
            if tuple(pretrained_table.shape) != tuple(self.dimensions):
                raise TypeError('\nDimensions of this Q-Table do not match dimensions of passed pre-trained table\n' +
                                'Pre-trained: ' + str(pretrained_table.shape) + '\n' +
                                'Q-Table: ' + str(tuple(self.dimensions)))

            self.q_table = pretrained_table

    # Given a continuous state, index the discrete q table and get the reward
    def get_rewards(self, state):
        # define the step range where continuous state values fall into discrete indices
        # for example if our dims is 10x10 and low = 0,0 high = 1,1 then steps = 0.1,0.1
        steps = (self.observation_space.high - self.observation_space.low) / self.dimensions[:-1] # disclude final dim which is action space
        # convert the n-dimensional state into an index in the Q table and return the rewards for all actions in state
        indices = (state - self.observation_space.low) / steps

        # get the reward
        return self.q_table[tuple(indices.astype(np.int))]

    # Given a continuous state, index the discrete q table and set the reward
    def set_reward(self, state, action, reward):
        # define the step range where continuous state values fall into discrete indices
        # for example if our dims is 10x10 and low = 0,0 high = 1,1 then steps = 0.1,0.1
        steps = (self.observation_space.high - self.observation_space.low) / self.dimensions[:-1] # disclude final dim which is action space
        # convert the n-dimensional state into an index in the Q table including the action and return the reward
        indices = (state - self.observation_space.low) / steps
        indices = np.array(list(indices) + [action])

        # set the reward
        self.q_table[tuple(indices.astype(np.int))] = reward


# Q Learning Agent for gym games with continous observation spaces and discrete action spaces
class QAgent:
    def __init__(self, env, disc_obsv_size, alpha=0.1, gamma=0.95, epsilon=0.5):
        self.env = env
        self.disc_obsv_size = disc_obsv_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = QTable(self.env, self.disc_obsv_size)

    # save this Agents Q Table for later use / training
    def save_table(self, file_name):
        with open(file_name, 'wb') as f:
            np.save(f, self.qtable.q_table)

    # load a saved Q Table from file
    def load_table(self, file_name):
        with open(file_name, 'rb') as f:
            pretrained_table = np.load(f)

        self.qtable = QTable(self.env, self.disc_obsv_size, pretrained_table)

    # play the game using the agent and qtable it generated
    def play_game(self):
        state = self.env.reset()

        done = False
        while not done:
            self.env.render()
            q_action = np.argmax(self.qtable.get_rewards(state))
            state, reward, info, done = self.env.step(q_action)

    # trains a q_table by approximating the continuous observation space on top of the discrete action space
    # in theory, increasing epochs(training time) and observation space(model size) will increase performance.
    def train(self, epochs, decay_interval=None, render_interval=1000, print_interval=50):
        if decay_interval is None:
            self.epsilon = 0.5  # random action threshold
        else:
            self.epsilon = 1  # random action threshold for decay

        # train over all the epochs
        for i in range(1, epochs+1):
            state = self.env.reset()

            # apply epsilon decay
            if decay_interval is not None and i % decay_interval == 0 and self.epsilon > 0:
                self.epsilon -= decay_interval / (epochs/2)

            # data for this epoch
            actions = {
                0: 0,
                1: 0,
                2: 0
            }
            done = False

            while not done:
                # render when specified
                if i % render_interval == 0:
                    self.env.render()

                # choose when to explore random actions based on threshold
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable.get_rewards(state))

                # make the action chosen above
                next_state, reward, info, done = self.env.step(action)

                # update the q table with the results of the action chosen
                old_value = self.qtable.get_rewards(state)[action]
                next_max = np.max(self.qtable.get_rewards(next_state))

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.qtable.set_reward(state, action, new_value)

                # update the state and epoch of the episode
                state = next_state
                # count the actions
                actions[action] += 1

            # print the state of the model
            if i % print_interval == 0:
                os.system('clear')
                print("Epoch:", i)
                print("Epsilon:", self.epsilon)
                print("Actions:", actions)
                print("Mean in Q-Table:", np.mean(self.qtable.q_table))
                print("Max in Q-Table:", np.max(self.qtable.q_table))
                print("Min in Q-Table:", np.min(self.qtable.q_table))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = QAgent(env, [20, 20])

    # ------------------------------- Training ----------------------------------
    agent.train(10000, decay_interval=100)
    agent.save_table('mountaincar.npy')

    # ------------------------------- Playing ------------------------------------
    agent.load_table('mountaincar.npy')
    agent.play_game()

