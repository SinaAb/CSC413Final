import retro
import gym
from baselines.common.retro_wrappers import *
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from collections import namedtuple, deque
from select import select
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time
import os
import sys


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

        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    # initialize the DQN agent which trains a convolution model given pixel data of the game
    def __init__(self, environment, instance_name, buffer_size=100000, batch_size=96, replay_every=64, target_update=None):
        # hyper parameters
        self.gamma = 0.987
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.994
        self.learning_rate = 0.000625

        self.env = environment  # game environment
        self.model = self.create_model()  # the DQN network
        self.instance_name = instance_name

        # we freeze the weights of this model and use it for Q predictions so that our gradient descent is not
        # stochastic. We will then incrementally update this model using the true model after some N training steps.
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())  # copy the true model

        # replay memory initialization
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.replay_every = replay_every  # number of training iterations taken before the target model is updated
        self.target_update = target_update # how often to update the target network
        self.time_step = 0  # tracks how many steps are taken

    def step(self, state, action, reward, next_state, done):
        # Save the experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # update the time_step tracker up to the replay limit
        self.time_step = (self.time_step + 1) % self.replay_every

        # train the true model through memory replay of the target model once replay_every time steps taken
        # update the target model to reflect the newly trained model
        if self.time_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.replay(experiences)

            # if target update is None then update the network every time a replay occurs otherwise update on the param
            if self.target_update is None:
                self.update_target_model()
            elif self.time_step % self.target_update == 0:
                self.update_target_model()

    # make an action choice
    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            batched_state = np.array([list(state)])
            qs = self.model.predict(batched_state)[0]

            # heuristic for selecting actions for a multi binary action space
            if type(self.env.action_space) == gym.spaces.MultiBinary:
                mean = np.mean(qs)
                best_action = np.array([int(q > mean) for q in qs])
                return best_action
            # classical q-learning discrete action selection
            elif type(self.env.action_space) == gym.spaces.Discrete:
                best_action = np.argmax(qs)
                return best_action

    def replay(self, experiences):
        s, a, r, ns, d = experiences

        for n in range(self.memory.batch_size):
            state, action, reward, next_state, done = s[n], a[n], r[n], ns[n], d[n]

            batched_next_state = np.array([list(next_state)]) # refer to below
            batched_state = np.array([list(state)])  # convert it into batch form (1, ... original shape ...)
            target_qs = self.target_model.predict(batched_state)[0]  # get the q values

            # set the q values for game ending actions. At worst negative or zero and at best a reward for winning
            # no need for predicting the next state to update q values as the game is now done or reset
            if done:
                # MultiBinary q update
                if type(self.env.action_space) == gym.spaces.MultiBinary:
                    for i in range(len(target_qs)):
                        target_qs[i] = reward if action[i] else target_qs[i]
                # Discrete q update
                elif type(self.env.action_space) == gym.spaces.Discrete:
                    target_qs[action] = reward

            # update the the qs with all the q values greater than the mean of the next_state
            # this heuristic is a key to solving this DQN for this action space, and will be updated regularly
            # traditionally we take the max best action for the next state and fit the model to it
            else:
                next_qs = self.target_model.predict(batched_next_state)[0]

                # Take the MultiBinary mean action heuristic.
                # This is a simplified heuristic from the continuous multi action space embedding paper. Much simpler us
                if type(self.env.action_space) == gym.spaces.MultiBinary:
                    mean_qs = np.mean(target_qs)

                    for i in range(len(target_qs)):
                        # instead of taking the max action on the next state take all actions that are above average
                        if target_qs[i] > mean_qs:
                            target_qs[i] = reward + next_qs[i] * self.gamma

                # take the classical best action from traditional Q-learning
                elif type(self.env.action_space) == gym.spaces.Discrete:
                    act_index = np.argmax(target_qs)
                    target_qs[act_index] = reward + next_qs[act_index] * self.gamma

            # fit the model on the target q values
            batched_target = np.array([list(target_qs)])
            self.model.fit(batched_state, batched_target, epochs=1, verbose=0)

    def train(self, num_episodes, max_t_steps):
        # for recording all scores and recent scores for analysis
        scores = []
        recent_scores = deque(maxlen=100)

        # iterate over num_episodes
        for n in range(num_episodes):
            # initialize episode info
            score = 0
            game_done = False
            final_time_step = 0
            state = env.reset()

            # run over max_t_steps so that the model doesnt run through the entire game losing with no good rewards
            for time_steps in range(max_t_steps):
                # get the q action from the model through the DQN Q policy. Epsilon decay and decision is handled here
                action = self.act(state)

                # take the action and record the results
                next_state, reward, done, _ = self.env.step(action)

                # saves the results into the replay buffer. update the time step counter and will call replay to train
                # the model iff self.update_every is reached. self.model is copied to target_model afterwards
                self.step(state, action, reward, next_state, done)

                # prepare for next iteration
                state = next_state
                score += reward

                # exit if the game is over
                if done:
                    game_done = done
                    final_time_step = time_steps
                    break

                os.system('clear')
                print("Time Step:", time_steps)

            # apply epsilon decay
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # record the scores
            scores.append(score)
            recent_scores.append(score)

            # print the system information
            os.system('clear')
            print("---------- Episode:", n, "---------- ")
            print("Score:", score)
            print("Epsilon:", self.epsilon)
            print("Time Steps:", final_time_step)
            print("Done:", game_done)
            print("Score Mean:", np.mean(np.array(list(recent_scores))))

            # give option to save the model on this episode, waits 3 seconds for input
            print("Enter any input to save the model and quit or wait 1.5 seconds...")
            timeout = 1.5
            rlist, _, _ = select([sys.stdin], [], [], timeout)
            if rlist:
                # save the model
                self.model.save(self.instance_name)
                # save the score
                with open('./' + self.instance_name + '/scores.npy', 'wb') as f:
                    np.save(f, np.array(scores))

                print("Saved and Quit")
                exit(1)

        # Save once training is complete
        # save the model
        self.model.save(self.instance_name)
        # save the score
        with open('./' + self.instance_name + '/scores.npy', 'wb') as f:
            np.save(f, np.array(scores))

        print("Saved and Completed Training")

    # copies the training models weights to the target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # copy the true model

    # Creates a CNN with two Conv-2D layers with ReLU
    # Model structure is taken from the Google DeepMind 2015 paper, possibly can improve with pooling and dropout
    def create_model(self):
        inputs = Input(shape=self.env.observation_space.shape)

        # Convolutions on the frames on the screen
        layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)
        layer4 = Flatten()(layer3)
        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(self.env.action_space.n)(layer5)

        model = keras.Model(inputs=inputs, outputs=action)
        model.compile(loss="mse", optimizer=Adam(lr=0.0003), metrics=['accuracy'])

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

            time.sleep(0.01633)

        env.close()

    # loads a previously trained model for further tweaking or playing the game
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.target_model.set_weights(self.model.get_weights())  # copy the true model

    # runs the game using the trained agent
    def q_run(self):
        state = self.env.reset()
        while True:
            self.env.render()
            action = self.act(state)
            new_state, rew, done, info = self.env.step(action)
            if done:
                break

            time.sleep(0.01633)

        self.env.close()


def wrap_environment_retro(env, instance_name):
    if instance_name != "":
        # create the directory to store the playbacks
        os.mkdir("./" + instance_name + "-playbacks")
        # save a video every k episodes
        env = MovieRecord(env, "./" + instance_name + "-playbacks", k=10)
        # Frame skip (hold an action for this many frames) and sticky actions
        env = StochasticFrameSkip(env, 4, 1)

    # scale and turn RGB image to grayscale
    env = WarpFrame(env, width=84, height=84, grayscale=True)

    return env

def wrap_environment_atari(env):
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = FrameStack(env, k=4)

    return env

if __name__ == '__main__':
    # ----------------- CODE FOR TRAINING -----------------------
    instance_name = "riverman2"

    env = gym.make('Riverraid-v0')
    env = wrap_environment_atari(env)

    agent = DQNAgent(env, instance_name, target_update=500)
    agent.train(num_episodes=400, max_t_steps=10000)

    # # ---------- for environment loading
    # agent.load_model('./skiboy420')

    # -------- plot stats
    # scores = np.load('./riverman/scores.npy')
    # steps = np.arange(1, len(scores) + 1)
    #
    # plt.plot(steps, scores)
    # plt.show()

    # ---------- play the trained model
    # agent.q_run()

    # ------------ Random Run game
    # env = gym.make('Riverraid-v0')
    #
    # env.reset()
    # steps = 0
    # while True:
    #     action = env.action_space.sample()
    #     new_state, rew, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         break
    #
    #     time.sleep(0.01633)
    #     steps += 1
    # print(steps)
    #
    # env.close()
