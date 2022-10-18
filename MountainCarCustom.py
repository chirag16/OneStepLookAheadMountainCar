"""
Created on Fri Oct  7 16:36:52 2022

@author: ctra
"""

'''
RL Project
'''

import gym
from MountainCarEnvironment import MountainCarEnvironment
import numpy as np
import matplotlib.pyplot as plt
from OneStepActorCritic import *

# np.random.seed(6)

# Initialize the Mountain Car Environment
env = MountainCarEnvironment()
state_space = env.observation_space
action_space = env.action_space


# Show some information about states and actions
print('Environment Details')
print('-' * 50)

# state_space = env.observation_space
print('state_space.low: {}'.format(state_space.low))
print('state_space.high: {}'.format(state_space.high))
n_features = state_space.shape[0]

action_space = env.action_space
print('action_space.n: {}'.format(action_space.n))
n_actions = action_space.n

# Run One Step Actor Critic on MountainCar

print('\n\n\n','*' * 50)
print('Training the Policy and State Evaluator using One-Step Actor Critic')
print('*' * 50, '\n\n\n')

num_runs = 4
iterations = 500

episode_lengths_history = np.zeros((num_runs, iterations))
theta_history = []
W_history = []

for run_no in range(num_runs):
    print('\nRun No.: {}\n'.format(run_no))
    theta, W, episode_lengths = os_actor_critic(env, sigma=1., gamma=1., order=10, iterations=iterations, \
                               alpha_theta=0.001, alpha_w=0.001, render=False)
    
    # Store Learnt Parameters
    theta_history.append(theta)
    W_history.append(W)
    
    # Remember episode lengths
    episode_lengths_history[run_no] = episode_lengths.copy()
    
    
# Choose best policy parameters according to mean episode length
episode_lengths_avg_per_policy = np.mean(episode_lengths_history, axis=0)
best_policy = np.argmax(episode_lengths_avg_per_policy)

theta = theta_history[best_policy]
W = W_history[best_policy]

# Calculate the average
episode_lengths_avg = np.mean(episode_lengths_history, axis=0)
episode_lengths_std = np.std(episode_lengths_history, axis=0)

# Plot the graph
plt.plot(np.arange(iterations), episode_lengths_avg)
plt.fill_between(np.arange(iterations), episode_lengths_avg - episode_lengths_std / 2, \
                 episode_lengths_avg + episode_lengths_std / 2, color='blue', alpha=0.2)
plt.title('Training: Avg. Episode Length v/s No. Episodes')
plt.xlabel('No. Episodes')
plt.ylabel('Avg. Episode Length')
plt.show()

# Run the learnt policy a few times and show its performance
print('\n\n\n','*' * 50)
print('Running Episodes Based on Offline Learnt Policy')
print('*' * 50, '\n\n\n')

to_display = 50
offline_lengths = np.zeros(to_display)

for episode_no in range(to_display):
    states, actions, rewards = generate_episode(env, theta, sigma=1., order=10, k=env.observation_space.shape[0], mode='sample', render=False)

    # Remember the episode length
    offline_lengths[episode_no] = len(states) - 1
    
    print('Episode: {},\tLength: {}'.format(episode_no, len(states) - 1))
    
# Run the learnt policy with One Step Look Ahead
print('\n\n\n','*' * 50)
print('Running Episodes Based on Online Policy')
print('*' * 50, '\n\n\n')

online_lengths = np.zeros(to_display)

for episode_no in range(to_display):
    states, actions, rewards = generate_osla_episode(env, theta, W, sigma=1., order=10, k=env.observation_space.shape[0], gamma=1., rollout_len=400, render=False)

    # Remember the episode length
    online_lengths[episode_no] = len(states) - 1

    print('Episode: {},\tLength: {}'.format(episode_no, len(states) - 1))

# Calculate and show the mean of episode lengths using the 2 approaches
print('Offline Policy Mean Episode Length: {:.4f}'.format(np.mean(offline_lengths)))
print('Online Policy Mean Episode Length: {:.4f}'.format(np.mean(online_lengths)))


# Terminate the environment
env.close()