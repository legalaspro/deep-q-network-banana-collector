import numpy as np
import random

from cnn_dueling_model import CNNDuelingQNetwork
from cnn_model import CNNQNetwork

from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import SumTreePrioritizedReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99         # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4         # learning rate
UPDATE_EVERY = 4        # how often to update the network

# Prioritize device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using CUDA.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using MPS.")
else:
    device = torch.device("cpu")
    print("Neither CUDA nor MPS is available. Using CPU.")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, name, 
                 learning_rate=LR, 
                 tau=TAU, 
                 gamma=GAMMA, 
                 batch_size=BATCH_SIZE,
                 buffer_size=BUFFER_SIZE,
                 is_double=False, 
                 is_dueling=False, 
                 is_prioritzed=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.is_double = is_double
        self.is_dueling = is_dueling
        self.is_prioritzed = is_prioritzed
        
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size

        self.beta = 0.4
        self.beta_increment = 1e-5
        

        # Q-Network - dueling or vanilla
        if is_dueling:
            self.qnetwork_local = CNNDuelingQNetwork(state_size, action_size).to(device)
            self.qnetwork_target = CNNDuelingQNetwork(state_size, action_size).to(device)
        else:
            self.qnetwork_local = CNNQNetwork(state_size, action_size).to(device)
            self.qnetwork_target = CNNQNetwork(state_size, action_size).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        if is_prioritzed:
            self.memory = SumTreePrioritizedReplayBuffer(action_size, buffer_size, self.batch_size, device)
        else:
            self.memory = ReplayBuffer(action_size, buffer_size, self.batch_size, device)
       
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        if self.is_prioritzed:
            self.memory.add(state, action, reward, next_state, done)

            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                if len(self.memory) >= self.batch_size: 
                    experiences, indices, weights = self.memory.sample(beta=self.beta)  # Sample with prioritized replay
                    self.learn_prioritized(experiences, indices, weights, self.gamma)

                    # Increment beta for better convergence
                    self.beta = min(1.0, self.beta + self.beta_increment)
        else: 

            self.memory.add(state, action, reward, next_state, done)
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)
 
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.is_double:
            self.run_double_dqn(experiences, gamma)
        else:
            self.run_dqn(experiences, gamma)
       
        # ------------------- update target network ------------------- #
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target)          
    

    # Implemented only for DDQN
    def learn_prioritized(self, experiences, indices, weights, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Double DQN logic for Q target calculation
        best_next_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get current Q-values
        Q_outputs = self.qnetwork_local(states).gather(1, actions)

        # Compute loss with IS weights
        td_errors = (Q_targets - Q_outputs).detach().cpu().numpy()  # For updating priorities

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * F.mse_loss(Q_outputs, Q_targets, reduction='none')).mean()
 
        # Backpropagation and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # so logically here we have both qnetwork_local and qnetwork_target in eval mode
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def run_double_dqn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # lets implament Double DQN here
        # Use the local network to select the best actions
        best_actions = self.qnetwork_local(next_states).argmax(dim=1).unsqueeze(1)
        # Get predicted Q Values (for next states) from target model using best actions from local network
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(dim=1, index=best_actions)
        # Compute Q targets for current states
        Q_targets = rewards + (1-dones)*gamma*Q_targets_next
        
        # Get expected Q values from local model
        Q_outputs = self.qnetwork_local(states).gather(dim=1, index=actions)

        loss = F.mse_loss(Q_outputs, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()   
        self.optimizer.step()
    
    def run_dqn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(dim=1)[0].unsqueeze(1) #to make shape (64,1)
        # compute target values (reward + gamma * max_action_values)
        # if done, target values is just reward
        Q_targets = rewards + ((1 - dones) * (gamma * Q_targets_next))

        # Get expected Q values from local model
        Q_outputs = self.qnetwork_local(states).gather(dim=1, index=actions)
        
        loss = F.mse_loss(Q_outputs, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()   
        self.optimizer.step()
    

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
