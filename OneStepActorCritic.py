'''
Rl Project
'''
import numpy as np
from nn_utils import *
from itertools import product

# Reuse combinations for fourier basis to save memory
combinations = None
combinations_generated = False

def policy_forward(features, theta, sigma=0.01):
    # Unpack parameters
    theta_w1 = theta['w1']
    theta_b1 = theta['b1']
    
    # print(theta_w1.shape)
    # print(features.shape)
    
    # Generate scores for each action
    scores, cache_1 = affine_forward(features, theta_w1, theta_b1)
    
    cache = (cache_1, sigma)
    
    # Generate probabilities for each action
    probs = softmax_forward(sigma * scores)
    
    return probs, cache

def policy_backward(action, probs, cache):
    # Unpack cache
    cache_1, sigma = cache
    
    # Begin backpropagation
    dscores = softmax_backward(action, probs) * sigma
    dfeatures, dtheta_w1, dtheta_b1 = affine_backward(dscores, cache_1)
    
    dtheta = {}
    dtheta['w1'] = dtheta_w1
    dtheta['b1'] = dtheta_b1
    
    return dfeatures, dtheta   


def value_forward(features, W, bias=True):
    # Unpack parameters
    W_w1 = W['w1']
    W_b1 = W['b1']
    
    # Generate scores for each action
    value, cache_1 = affine_forward(features, W_w1, W_b1)
    
    cache = cache_1
    
    return value, cache

def value_backward(cache):
    # Unpack cache
    cache_1 = cache
    
    # Begin backpropagation
    dvalue = np.ones((1,))
    dfeatures, dW_w1, dW_b1 = affine_backward(dvalue, cache_1)
    
    dW = {}
    dW['w1'] = dW_w1
    dW['b1'] = dW_b1
    
    return dfeatures, dW  


def get_action(features, theta, sigma, mode='sample'):
    # Calculate probabilities
    probs, _ = policy_forward(features, theta, sigma)
    
    # Sample an action based on probabities
    if mode == 'sample':
        n_actions = probs.shape[0]
        action = np.random.choice(n_actions, 1, p=probs)[0]
        
        return action
    
    # Return max probability action
    action = np.argmax(probs)
    return action

def get_osla_action(env, theta, W, sigma, gamma, order, k, rollout_len=300):
    # Since we will be modifying the state to simulate lookahead,
    # remember the current state so that we can reset the environment back to this state
    # once we have chosen the action
    S_0 = env.get_state()
    
    n_actions = env.action_space.n
    
    # Store the Quality of each Action
    Q = np.zeros(n_actions)
    
    for action in range(n_actions):
        # Exact One Step Look Ahead
        X_t, v_t = env.transition(action)
        S_t = (X_t, v_t)
        env.set_state(S_t)
        
        t = env.t + 1
        
        done = env.check_termination(S_t, t)
        
        R_t = -1
        if X_t == env.X_max:
            R_t = 0
        
        # Initialized the Current Action's Quality
        Q[action] = R_t
        
        # Truncated Rollout
        # i.e. use the offline learnt policy to transition
        roll_no = 0
        while not done and roll_no < rollout_len:
            # Get an action depending on the policy
            features = get_features(env, S_t, order, k)
            a_t = get_action(features, theta, sigma, mode='max')
    
            # Perform the action
            X_tp1, v_tp1 = env.transition(a_t)
            S_tp1 = (X_tp1, v_tp1)
            env.set_state(S_tp1)
            
            t += 1
        
            done = env.check_termination(S_tp1, t)
            
            R_t = -1
            if X_t == env.X_max:
                R_t = 0
            
            # Update Action Quality
            Q[action] += gamma ** (roll_no + 1) * R_t
            
            # Update state
            S_t = S_tp1
            roll_no += 1
            
            
        # Calculate the value of the state reached at the end of the rollout
        # using the offline trained state evaluator and add it to the Action Quality
        features = get_features(env, S_t, order, k)
        Q[action] += gamma ** (roll_no) * value_forward(features, W)[0]
        
        # Reset the state
        env.set_state(S_0)
        
    # Get the action that maximizes the probability
    action = np.argmax(Q)
    
    return action

def get_features(env, state, order=5, k=4):
    global combinations, combinations_generated
    
    # Clipped State
    low = env.observation_space.low                   # Comment these is using Cart Pole
    high = env.observation_space.high
    clipped = np.clip(state, low, high)
    
    # Normalize the state
    normalized = (clipped - low) / (high - low)
    
    # Compute Fourier-Basis features
    if combinations_generated == False:
        combinations = np.array(list(product(range(order + 1), repeat=k)))
        combinations_generated = True
        
    features = np.cos(np.pi * np.matmul(combinations, np.expand_dims(normalized, axis=1)))
    
    return np.squeeze(features)

def generate_episode(env, theta, sigma, order, k, mode='sample', render=False):
    # Reset the position
    S_t = env.reset()
    done = False
    
    # Initialize the lists store observations from the episode
    states = [S_t]
    actions = []
    rewards = []
    
    # Loop till episode ends
    while not done:
        # Get an action depending on the policy
        features = get_features(env, S_t, order, k)
        a_t = get_action(features, theta, sigma, mode=mode)
        
        # Perform the action
        S_tp1, R_t, done, _ = env.step(a_t)
        
        # Store the observations
        states.append(S_tp1)
        actions.append(a_t)
        rewards.append(R_t)
        
        # Update state
        S_t = S_tp1
        
        # Render if requested
        if render == True:
            env.render()
        
    return states, actions, rewards

def generate_osla_episode(env, theta, W, sigma, order, k, gamma, rollout_len, render=False):
    # Reset the position
    S_t = env.reset()
    done = False
    
    # Initialize the lists store observations from the episode
    states = [S_t]
    actions = []
    rewards = []
    
    # Loop till episode ends
    while not done:
        # Get an action depending on the policy
        # features = get_features(env, S_t, order, k)
        # a_t = get_action(features, theta, sigma, mode='max')
        
        # Get an action depending on the policy
        a_t = get_osla_action(env, theta, W, sigma, gamma, order, k, rollout_len=rollout_len)
        
        # Perform the action
        S_tp1, R_t, done, _ = env.step(a_t)
        
        # Store the observations
        states.append(S_tp1)
        actions.append(a_t)
        rewards.append(R_t)
        
        # Update state
        S_t = S_tp1
        
        # Render if requested
        if render == True:
            env.render()
        
    return states, actions, rewards
    

# One-Step Actor Critic
def os_actor_critic(env, sigma=1., order=5, gamma=0.99, iterations=2000, \
                    alpha_w=0.05, alpha_theta=0.0008, print_every=20, \
                    render=True):
    # State space length
    s_space = env.observation_space
    k = s_space.shape[0] 
    n_features = (order + 1) ** k # Fourier Basis Features
    
    # Get the number of actions
    a_space = env.action_space
    n_actions = a_space.n
    
    # Initialize the policy parameters
    theta_w1 = np.random.randn(n_features, n_actions)
    theta_b1 = np.zeros((n_actions,))
    
    theta = {'w1': theta_w1, 'b1': theta_b1}
    
    # Initialize the value function parameters
    W_w1 = np.random.randn(n_features, 1)
    W_b1 = np.zeros((1,))
    
    W = {'w1': W_w1, 'b1': W_b1}
    
    # Store episode lengths for plotting
    episode_lengths = []
    
    # Loop over episodes
    for iteration in range(iterations):
        # Generate episodes
        S_t = env.reset()
        done = False
        
        T = 0
        while not done:
            if render == True:
                env.render()
            
            # Get an action depending on the policy
            f_t = get_features(env, S_t, order, k)
            a_t = get_action(f_t, theta, sigma)
            
            # Perform the action
            S_tp1, R_t, done, _ = env.step(a_t)
        
            # Calculate the TD-error
            v_t, cache_W = value_forward(f_t, W)
            
            f_tp1 = get_features(env, S_tp1, order, k)
            v_tp1, _ = value_forward(f_tp1, W)
            
            delta = R_t + gamma * v_tp1 - v_t
            
            # Calculate gradient wrt W
            _, dW = value_backward(cache_W)
            
            # Update critic  
            W['w1'] += alpha_w * delta * dW['w1']
            W['b1'] += alpha_w * delta * dW['b1']
            
            # Calculate the gradient wrt theta
            probs, cache = policy_forward(f_t, theta, sigma)
            _, dtheta = policy_backward(a_t, probs, cache)
            
            # Update policy parameters
            theta['w1'] += alpha_theta * delta * dtheta['w1']
            theta['b1'] += alpha_theta * delta * dtheta['b1']
            
            # Update the state
            S_t = S_tp1
        
            # Increment episode length
            T += 1
            
        # Store episode lengths
        episode_lengths.append(T) # Store this for plotting
            
        # Display episode length every few episodes
        if iteration % print_every == 0 or iteration == iterations - 1:
            print('Iteration: {}\tEpisode Length: {}'.format(iteration, T))
            
    return theta, W, np.array(episode_lengths)          
            
            
            
            
            
            
            
            
            
            
            