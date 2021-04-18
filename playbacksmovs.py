import retro
import os
import numpy as np
import time


def get_file_code(playback):
    p_num = "000000" + str(playback)
    return p_num[-6:]


def analyze_runs():
    scores = []
    timesteps = []

    for playback in range(len(os.listdir('./improved-memory2-playbacks/'))):
        filename = './improved-memory2-playbacks/Airstriker-Genesis-Level1-{}.bk2'.format(get_file_code(playback))
        print(filename)

        movie = retro.Movie(filename)  # 31 best performance so far for mean-action
        movie.step()
        env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players,)
        env.initial_state = movie.get_state()
        env.reset()

        score = 0
        t = 0
        while movie.step():
            keys = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    keys.append(movie.get_key(i, p))
            state, reward, done, info = env.step(keys)

            t += 1
            score += reward

        scores.append(score)
        timesteps.append(t)

        env.close()
        movie.close()

    scores = np.array(scores)
    timesteps = np.array(timesteps)

    print("Average Score:", np.mean(scores))
    print("Average Timesteps:", np.mean(timesteps))
    print("Max Score:", np.max(scores), "On Playback:", np.argmax(scores))
    print("Max Timesteps:", np.max(timesteps), "On Playback:", np.argmax(timesteps))


def run_playback():
    filename = './improved-memory2-playbacks/Airstriker-Genesis-Level1-000164.bk2'
    print(filename)

    movie = retro.Movie(filename)  # 31 best performance so far for mean-action
    movie.step()
    env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL,players=movie.players,)
    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        env.render()

        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        env.step(keys)

        time.sleep(0.01)

    env.close()
    movie.close()


if __name__ == '__main__':
    run_playback()