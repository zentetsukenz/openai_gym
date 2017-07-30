import random as r
import collections as cl
import numpy as np
import gym
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_agent(
        n_state,
        n_action,
        learning_rate=0.001,
        discount_rate=0.999,
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
    model.add(Dense(16, input_dim=agent['n_state'], activation='relu'))
    model.add(Dense(8, activation='relu'))
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

def replay(agent, model, target_model, memory, batch_size):
    minibatch = r.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = target_model.predict(state)
        if done:
            target[0][action] = reward
        else:
            action_to_use = np.argmax(model.predict(next_state)[0])
            target[0][action] = reward + agent['discount_rate'] * target_model.predict(next_state)[0][action_to_use]
        model.fit(state, target, epochs=1, verbose=0)
    if agent['exploration_rate'] > agent['exploration_min']:
        agent['exploration_rate'] *= agent['exploration_decay_rate']
    return agent, model, target_model

n_episode = 5000
max_step = 200
replay_batch_size = 32

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './cartpole-experiment-1')

agent = build_agent(env.observation_space.shape[0], env.action_space.n)
memory = build_memory(maxlen=2000)
model = build_model(agent)
target_model = build_model(agent)

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
        agent, model, target_model = replay(agent, model, target_model, memory, replay_batch_size)

    if i_episode % 30 == 0:
        target_model.set_weights(model.get_weights())
