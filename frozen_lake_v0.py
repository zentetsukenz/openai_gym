import gym
import numpy as np

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
learning_rate = .8
discount_rate = .95
num_episodes = 2000

rewards = []
for i in range(num_episodes):
  # Reset environment and get first new observation
  state = env.reset()
  cumulative_reward = 0
  d = False
  j = 0

  # The Q-Table learning algorithm
  while j < 99:
    j += 1

    # Choose an action by greedily (with noise) picking from Q table
    action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

    # Get new state and reward from environment
    next_state, reward, is_finish, _ = env.step(action)

    # Update Q-Table with new knowledge
    Q[state, action] = Q[state, action] + learning_rate * (reward + discount_rate *
            np.max(Q[next_state, :]) - Q[state, action])

    cumulative_reward += reward
    state = next_state

    if is_finish == True:
      break

  rewards.append(cumulative_reward)

print("Score over time: " +  str(sum(rewards) / num_episodes))
print("Final Q-Table Values")
print(Q)
