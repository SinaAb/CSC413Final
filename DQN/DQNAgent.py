import retro
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import namedtuple, deque
import numpy as np
import random
import time


# Custom tensorboard class for log writing so we dont have 10 million logs haha
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

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
        self.env = environment  # game environment
        self.model = self.create_model()  # the DQN network
        self.model_name = "DQN: " + self.env.gamename + "\nSize: " + str(self.env.observation_space.shape)

        # we freeze the weights of this model and use it for Q predictions so that our gradient descent is not
        # stochastic. We will then incrementally update this model using the true model after some N training steps.
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())  # copy the true model

        # replay memory initialization
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.replay_every = replay_every # number of training iterations taken before the target model is updated
        self.time_step = 0

        # our TensorBoard for writing to one log for every time we call model.fit()
        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{self.model_name}-{int(time.time())}')

        # hyper parameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

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

    def train(self, experiences):


    # Creates a CNN with two CONV2D layers with ReLU, Pooling, and Dropout.
    # Data is finally passed through 2 Dense layers which outputs a sigmoid activated output of action probabilities
    def create_model(self):
        model = keras.Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(self.env.action_space.n, activation='sigmoid'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def get_action_meanings(self):
        meaning = ""
        for i in range(self.env.action_space.n):
            act = np.zeros(self.env.action_space.n)
            act[i] = 1

            meaning += "Action: " + str(act) + " Meaning: " + str(self.env.get_action_meaning(act)) + "\n"

        return meaning


def random_run(self, env):
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
    # env = retro.make(game='Airstriker-Genesis')
    #
    # agent = DQNAgent(env)
    # agent.random_run()
