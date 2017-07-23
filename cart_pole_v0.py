import pandas as pd
import numpy as np
import gym
from gym import wrappers

def build_state(observation, feature_bins):
    return_val = ""
    for i in range(len(feature_bins)):
        return_val = return_val + str(np.digitize([observation[i]], feature_bins[i])[0])
    return int(return_val)

def build_feature_bins(lows, highs, num_bin):
    return np.array([
        pd.cut([lows[0], highs[0]], bins=num_bin, retbins=True)[1][1:-1],
        pd.cut([lows[1], highs[1]], bins=num_bin, retbins=True)[1][1:-1],
        pd.cut([lows[2], highs[2]], bins=num_bin, retbins=True)[1][1:-1],
        pd.cut([lows[3], highs[3]], bins=num_bin, retbins=True)[1][1:-1],
    ])

def update_q_table(
        q_table,
        state,
        action,
        state_prime,
        action_prime,
        reward,
        learning_rate=0.2,
        discount_rate=0.9):
    q_table[state, action] = \
            (1 - learning_rate) * q_table[state, action] \
            + learning_rate \
            * (reward + discount_rate * q_table[state_prime, action_prime])
    return q_table

def select_action(
        q_table,
        state,
        num_action,
        exploration_rate=0.5,
        exploration_rate_decay=0.99):
    if (1 - exploration_rate) <= np.random.uniform(0, 1):
        return np.random.randint(0, num_action), exploration_rate * exploration_rate_decay
    else:
        return q_table[state].argsort()[-1], exploration_rate * exploration_rate_decay

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './cartpole-experiment')

num_epoch = 5000
num_feature = env.observation_space.shape[0]
num_action = env.action_space.n
num_bin = 10

learning_rate = 0.3
discount_rate = 1
exploration_rate = 1
exploration_rate_decay = 0.999

feature_bins = build_feature_bins(
    env.observation_space.low,
    env.observation_space.high,
    num_bin
)

q_table = np.zeros((num_bin ** num_feature, num_action))

for i_episode in range(num_epoch):
    observation = env.reset()

    for t in range(200):
        env.render()

        state = build_state(observation, feature_bins)
        action, exploration_rate = select_action(q_table,
                state,
                num_action,
                exploration_rate=exploration_rate,
                exploration_rate_decay=exploration_rate_decay)

        observation_prime, reward, done, info = env.step(action)

        state_prime = build_state(observation_prime, feature_bins)
        action_prime, exploration_rate = select_action(
                q_table,
                state_prime,
                num_action,
                exploration_rate=exploration_rate,
                exploration_rate_decay=exploration_rate_decay)

        if done:
            if t < 195:
                reward = -100000
            else:
                reward = 1000
        else:
            reward += t

        q_table = update_q_table(
            q_table,
            state,
            action,
            state_prime,
            action_prime,
            reward,
            learning_rate=learning_rate,
            discount_rate=discount_rate)

        if done:
            break

        observation = observation_prime

    print("Done episode = {}, t = {}".format(i_episode, t + 1))
