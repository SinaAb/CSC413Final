import retro
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from collections import namedtuple, deque
import numpy as np
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
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save the experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # update the target model once we reach replay_every time steps taken
        self.time_step = (self.time_step + 1) % self.replay_every
        if self.time_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            pass

    def train(self, experiences):
        pass

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
        # Network defined by the Deepmind paper
        inputs = Input(shape=(224, 320, 3))

        # Convolutions on the frames on the screen
        layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = Flatten()(layer3)

        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(12, activation="sigmoid")(layer5)

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


def random_run(env):
    env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break

        if rew > 0:
            print("Reward:", rew)

        time.sleep(0.0166)
    env.close()

if __name__ == '__main__':
    env = retro.make(game='Airstriker-Genesis')

    agent = DQNAgent(env)
    print(agent.model_name)
