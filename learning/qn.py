import random as r
import pandas as pd
import collections as cl
import numpy as np

def build_state(agent, state):
    return_val = ""
    feature_bins = agent['feature_bins']
    for i in range(len(feature_bins)):
        return_val += str(np.digitize([state[i]], feature_bins[i])[0])
    return int(return_val)

def build_agent(
        environment,
        learning_rate=0.001,
        discount_rate=0.999,
        exploration_rate=1.,
        exploration_decay_rate=0.995,
        exploration_min=0.01,
        num_bin=10):

    feature_bins = __build_feature_bins(
        environment.observation_space.low,
        environment.observation_space.high,
        num_bin
    )

    return {
        'n_state':                environment.observation_space.shape[0],
        'n_action':               environment.action_space.n,
        'learning_rate':          learning_rate,
        'discount_rate':          discount_rate,
        'exploration_rate':       exploration_rate,
        'exploration_decay_rate': exploration_decay_rate,
        'exploration_min':        exploration_min,

        'feature_bins':           feature_bins,
        'n_bin':                  num_bin,
    }

def build_memory(maxlen=2000):
    return cl.deque(maxlen=maxlen)

def remember(memory, state, action, reward, next_state, done):
    new_memory = cl.deque(memory)
    new_memory.append((state, action, reward, next_state, done))
    return new_memory

def act(agent, model, state):
    if np.random.rand() <= agent['exploration_rate']:
        return r.randrange(agent['n_action'])
    return np.argmax(model['online'][state])

def build_model(agent, loss_function='mse'):
    return {'online': __build_model(agent)}

def replay(agent, model, memory, batch_size):
    online_model = model['online']
    learning_rate = agent['learning_rate']
    discount_rate = agent['discount_rate']

    state, action, reward, next_state, done = memory.popleft()
    next_action = np.argmax(online_model[next_state])

    online_model[state, action] = \
            (1 - learning_rate) * online_model[state, action] \
            + learning_rate \
            * (reward + discount_rate * online_model[next_state, next_action])

    if agent['exploration_rate'] > agent['exploration_min']:
        agent['exploration_rate'] *= agent['exploration_decay_rate']

    return agent, {'online': online_model}, memory

def update_target(agent, model):
    return agent, model

def __build_model(agent):
    n_bin     = agent['n_bin']
    n_state = agent['n_state']
    n_action  = agent['n_action']
    return np.zeros((n_bin ** n_state, n_action))

def __build_feature_bins(lows, highs, num_bin):
    arr = np.array([])
    for i in len(lows):
        arr = np.append(arr, pd.cut(
            [lows[i], highs[i]],
            bins=num_bin,
            retbins=True)[1][1:-1])
    return arr
