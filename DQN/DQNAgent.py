import retro
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from collections import namedtuple, deque
import numpy as np
import tensorflow as tf
import random
import time


# the Replay Memory for training the target network
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array(e.state for e in experiences if e is not None)
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    # initialize the DQN agent which trains a convolution model given pixel data of the game
    def __init__(self, environment, buffer_size=50000, batch_size=128, replay_every=32):
        # hyper parameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

        self.env = environment  # game environment
        self.model = self.create_model()  # the DQN network
        self.model_name = "DQN: " + self.env.gamename + "\nSize: " + str(self.env.observation_space.shape)

        # we freeze the weights of this model and use it for Q predictions so that our gradient descent is not
        # stochastic. We will then incrementally update this model using the true model after some N training steps.
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())  # copy the true model

        # replay memory initialization
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.replay_every = replay_every  # number of training iterations taken before the target model is updated
        self.time_step = 0  # tracks how many steps are taken

    def step(self, state, action, reward, next_state, done):
        # Save the experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # update the time_step tracker up to the replay limit
        self.time_step = (self.time_step + 1) % self.replay_every

        # update the target model once we reach replay_every time steps taken
        if self.time_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.replay(experiences)

    # make an action choice
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            batched_state = np.array([list(state)])
            qs = self.model.predict(batched_state)[0]
            mean = np.mean(qs)
            best_action = np.array([int(q > mean) for q in qs])
            return best_action

    def replay(self, experiences):
        for sample in experiences:
            state, action, reward, next_state, done = sample

            batched_next_state = np.array([list(next_state)]) # refer to below
            batched_state = np.array([list(state)])  # convert it into batch form (1, ... original shape ...)
            target_qs = self.target_model.predict(batched_state)[0]  # get the q values

            # set the q values for the actions taken to the reward otherwise keep them the same
            if done:
                for i in range(len(target_qs)):
                    target_qs[i] = reward if action[i] else target_qs[i]

            # update the the qs with all the q values greater than the mean of the next_state
            # this heuristic is a key to solving this DQN for this action space, and will be updated regularly
            # traditionally we take the max best action for the next state and fit the model to it
            else:
                mean_qs = np.mean(target_qs)
                next_qs = self.target_model.predict(batched_next_state)[0]

                for i in range(len(target_qs)):
                    # instead of taking the max action on the next state take all actions that are above average for now
                    if target_qs[i] > mean_qs:
                        target_qs[i] = reward + next_qs[i] * self.gamma

            # fit the model on the target q values
            batched_target = np.array([list(target_qs)])
            self.model.fit(state, batched_target, epochs=1, verbose=0)

    # copies the training models weights to the target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # copy the true model

    # Creates a CNN with two CONV2D layers with ReLU, Pooling, and Dropout.
    # Data is finally passed through 2 Dense layers which outputs a sigmoid activated output of action probabilities
    def create_model(self):
        # model = keras.Sequential()
        #
        # model.add(Conv2D(256, (3, 3), input_shape=self.env.observation_space.shape))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        #
        # model.add(Conv2D(256, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        #
        # model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        # model.add(Dense(64))
        #
        # model.add(Dense(self.env.action_space.n, activation='sigmoid'))
        # model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        # Gym Retro customized Convolution DQN model from DeepMind (big thanks to the authors :D)
        inputs = Input(shape=self.env.observation_space.shape)

        # Convolutions on the frames on the screen
        layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)
        layer4 = Flatten()(layer3)
        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(self.env.action_space.n)(layer5)

        model = keras.Model(inputs=inputs, outputs=action)
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model

    def get_action_meanings(self):
        meaning = ""
        for i in range(self.env.action_space.n):
            act = np.zeros(self.env.action_space.n)
            act[i] = 1

            meaning += "Action: " + str(act) + " Meaning: " + str(self.env.get_action_meaning(act)) + "\n"

        return meaning

    def print_model(self):
        self.model.summary()

    def random_run(self):
        env = self.env
        env.reset()
        while True:
            action = env.action_space.sample()
            new_state, rew, done, info = env.step(action)
            env.render()
            if done:
                break

            if rew > 0:
                qs = self.target_model.predict(np.array([list(new_state)]))
                print(qs)
                print("mean:", np.mean(qs))
                print("max:", np.max(qs))

            time.sleep(0.01633)

        env.close()


if __name__ == '__main__':
    env = retro.make(game='Airstriker-Genesis')

    agent = DQNAgent(env)
    print(agent.get_action_meanings())
    agent.random_run()

