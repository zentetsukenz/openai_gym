import numpy as np
import gym
import learning.ddqn as learner
from gym import wrappers

n_episode = 5000
max_step = 200
replay_batch_size = 32

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './cartpole-experiment-1')

agent  = learner.build_agent(env)
memory = learner.build_memory(maxlen=2000)
model  = learner.build_model(agent)

for i_episode in range(n_episode):
    state = learner.build_state(agent, env.reset())

    for t in range(max_step):
        env.render()

        action = learner.act(agent, model, state)

        next_state, reward, done, _info = env.step(action)
        next_state = learner.build_state(agent, next_state)

        memory = learner.remember(memory, state, action, reward, next_state, done)

        state = next_state

        agent, model = learner.replay(agent, model, memory, replay_batch_size)
        agent, model = learner.update_target(agent, model)

        if done:
            print("Done episode {}, t = {}".format(i_episode + 1, t + 1))
            break
