import random as r
import collections as cl
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_state(agent, state):
    return np.reshape(state, (1, agent['n_state']))

def build_agent(
        environment,
        learning_rate=0.001,
        discount_rate=0.999,
        exploration_rate=1.,
        exploration_decay_rate=0.995,
        exploration_min=0.01,
        model_update_rate=0.001,
        count_until_model_update=0,
        model_replay_rate=0.001,
        count_until_model_replay=0):

    return {
        'n_state':                  environment.observation_space.shape[0],
        'n_action':                 environment.action_space.n,

        'learning_rate':            learning_rate,
        'discount_rate':            discount_rate,
        'exploration_rate':         exploration_rate,
        'exploration_decay_rate':   exploration_decay_rate,
        'exploration_min':          exploration_min,

        'model_update_rate':        model_update_rate,
        'count_until_model_update': count_until_model_update,

        'model_replay_rate':        model_replay_rate,
        'count_until_model_replay': count_until_model_replay,
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
    return np.argmax(model['online'].predict(state)[0])

def build_model(agent, loss_function='mse'):
    return {'online': __build_model(agent), 'target': __build_model(agent)}

def replay(agent, model, memory, batch_size):
    if agent['count_until_model_replay'] < agent['model_replay_rate'] * 100000:
        agent['count_until_model_replay'] += 1
        return agent, model, memory

    if len(memory) <= batch_size:
        return agent, model

    online_model = model['online']
    target_model = model['target']

    minibatch = r.sample(memory, batch_size)

    for state, action, reward, next_state, done in minibatch:
        target = online_model.predict(state)

        if done:
            target[0][action] = reward
        else:
            action_to_use = np.argmax(online_model.predict(next_state)[0])
            target_value = target_model.predict(next_state)[0][action_to_use]
            target[0][action] = reward + agent['discount_rate'] * target_value

        online_model.fit(state, target, epochs=1, verbose=0)

    if agent['exploration_rate'] > agent['exploration_min']:
        agent['exploration_rate'] *= agent['exploration_decay_rate']

    return agent, {'online': online_model, 'target': target_model}, memory

def update_target(agent, model):
    if agent['count_until_model_update'] < agent['model_update_rate'] * 100000:
        agent['count_until_model_update'] += 1
        return agent, model

    agent['count_until_model_update'] = 0
    target_model = model['target']
    online_model = model['online']

    target_model.set_weights(online_model.get_weights())

    return {'online': online_model, 'target': target_model}

def __build_model(agent, loss_function='mse'):
    model = Sequential()
    model.add(Dense(32, input_dim=agent['n_state'], activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(agent['n_action'], activation='linear'))
    model.compile(loss=loss_function, optimizer=Adam(lr=agent['learning_rate']))

    return model
