'''
CS 687 - Reinforcement Learning
FALL 2021

Homework - 2

Author:     Chirag Trasikar
Spire ID:   32288082
'''
import numpy as np

class Environment:
    def __init__(self):
        pass
    
    def step(self, a_t):
        pass
    
    def reset(self):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    
class ObservationSpace:
    def __init__(self, low, high, shape):
        self.low = np.array(low).astype(np.float32)
        self.high = np.array(high).astype(np.float32)
        self.shape = shape
        
class ActionSpace:
    def __init__(self, n, action_set):
        self.n = n
        self.action_set = action_set

class MountainCarEnvironment(Environment):
    def __init__(self):
        # Set the minimum and maximum positions
        self.X_min = -1.2
        self.X_max = 0.5
        self.v_min = -0.07
        self.v_max = 0.07
        
        # State and Action Spaces
        self.observation_space = ObservationSpace(low=[self.X_min, self.v_min], high=[self.X_max, self.v_max], shape=[2])
        self.action_space = ActionSpace(n=3, action_set=[-1, 0, 1])
        
        # Set the max no. of timesteps per episode
        self.T = 400
        
        # Initialize the rest of the environment
        self.reset()
    
    def reset(self, X_range=(-0.6, -0.4), v_range=(0, 0)):
        # Sample initial X position X_0
        X_0 = np.random.uniform(X_range[0], X_range[1])
        v_0 = np.random.uniform(v_range[0], v_range[1])
        
        # Set the initial state
        self.S_t = (X_0, v_0)
        
        # Reset the time
        self.t = 0
        
        # Return the initial state
        return self.S_t
    
    def get_state(self):
        return self.S_t
    
    def set_state(self, S_t):
        self.S_t = S_t
    
    def transition(self, a_t):
        # Get the current position and velocity
        X_t, v_t = self.S_t
        
        # Calculate the new velocity
        v_tp1 = v_t + 0.001 * self.action_space.action_set[a_t] - 0.0025 * np.cos(3 * X_t)
        
        # Calculate the new position
        X_tp1 = X_t + v_tp1
        
        # Clip the velocity
        v_tp1 = min(max(v_tp1, self.v_min), self.v_max)
        
        # Clip the position
        X_tp1 = min(max(X_tp1, self.X_min), self.X_max)
        
        # Reset velocity if the car collided with walls
        if X_tp1 == self.X_min or X_tp1 == self.X_max:
            v_tp1 = 0
            
        return (X_tp1, v_tp1)
    
    def check_termination(self, S_t, t):
        done = False
        if S_t[0] == self.X_max or t == self.T:
            done = True
            
        return done
    
    def step(self, a_t):
        # Execute the action and Update the state
        X_tp1, v_tp1 = self.transition(a_t)
        S_tp1 = (X_tp1, v_tp1)
        self.S_t = S_tp1
        
        # Compute the reward
        R_t = -1
        if X_tp1 == self.X_max:
            R_t = 0
            
        # Update the time
        self.t += 1
        
        # Check for termination
        done = self.check_termination(S_tp1, self.t)
        
        # Return the new state and the reward
        return S_tp1, R_t, done, None
    
    def normalize_state(self, S_t, mode='cos'):
        # Unpack the state
        X_t, v_t = S_t
        
        # Initialize the normalized state
        X_t_norm, v_t_norm = None, None
        
        # Normalize between 0 and 1 if fourier-basis transformation uses cos
        if mode == 'cos':
            X_t_norm = 1 * (X_t - self.X_min) / (self.X_max - self.X_min)
            v_t_norm = 1 * (v_t - self.v_min) / (self.v_max - self.v_min)
        else:   # Normalize between -1 and 1 otherwise
            X_t_norm = -1 + 2 * (X_t - self.X_min) / (self.X_max - self.X_min)
            v_t_norm = -1 + 2 * (v_t - self.v_min) / (self.v_max - self.v_min)
            
        return X_t_norm, v_t_norm
    
    def render(self):
        print('T: {},\tX: {:.4f},\tV: {:.4f}'.format(self.t, self.S_t[0], self.S_t[1]))
        
        