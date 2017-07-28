import random as r
import collections as cl
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_agent(
        n_state,
        n_action,
        learning_rate=0.001,
        discount_rate=0.95,
        exploration_rate=1.,
        exploration_decay_rate=0.995,
        exploration_min=0.01):
    return {
        'n_state':                n_state,
        'n_action':               n_action,
        'learning_rate':          learning_rate,
        'discount_rate':          discount_rate,
        'exploration_rate':       exploration_rate,
        'exploration_decay_rate': exploration_decay_rate,
        'exploration_min':        exploration_min,
    }

def build_memory(maxlen=2000):
    return cl.deque(maxlen=maxlen)

def build_model(agent, loss_function='mse'):
    model = Sequential()
    model.add(Dense(48, input_dim=agent['n_state'], activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(agent['n_action'], activation='linear'))
    model.compile(loss=loss_function, optimizer=Adam(lr=agent['learning_rate']))
    return model

def remember(memory, state, action, reward, next_state, done):
    new_memory = cl.deque(memory)
    new_memory.append((state, action, reward, next_state, done))
    return new_memory

def act(agent, model, state):
    if np.random.rand() <= agent['exploration_rate']:
        return r.randrange(agent['n_action'])
    return np.argmax(model.predict(state)[0])

def replay(agent, model, memory, batch_size):
    minibatch = r.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(state)
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + agent['discount_rate'] * np.amax(model.predict(next_state)[0])
        model.fit(state, target, epochs=1, verbose=0)
    if agent['exploration_rate'] > agent['exploration_min']:
        agent['exploration_rate'] *= agent['exploration_decay_rate']
    return agent, model

n_episode = 5000
max_step = 200
replay_batch_size = 64

env = gym.make('CartPole-v0')

agent = build_agent(env.observation_space.shape[0], env.action_space.n)
memory = build_memory(maxlen=2000)
model = build_model(agent)

for i_episode in range(n_episode):
    state = env.reset()
    state = np.reshape(state, (1, agent['n_state']))

    for t in range(max_step):
        env.render()

        action = act(agent, model, state)

        next_state, reward, done, _info = env.step(action)
        next_state = np.reshape(next_state, (1, agent['n_state']))

        memory = remember(memory, state, action, reward, next_state, done)

        state = next_state

        if done:
            print("Done episode {}, t = {}".format(i_episode + 1, t + 1))
            break

    if len(memory) > replay_batch_size:
        agent, model = replay(agent, model, memory, replay_batch_size)
