import numpy as np
import random
import torch
import pickle

from collections import namedtuple

# Named tuple for storing experience tuples
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class SumTreePrioritizedReplayBuffer:
    """A prioritized replay buffer using a Sum Tree for efficient sampling."""

    def __init__(self, action_size, buffer_size, batch_size, device, alpha=0.5, max_priority=True):
        """
        Initialize a PrioritizedReplayBuffer with Sum Tree for efficient prioritized sampling.
        
        Args:
            action_size (int): Dimension of each action.
            buffer_size (int): Maximum size of the buffer.
            batch_size (int): Size of each training batch.
            device (torch.device): Device to perform tensor operations.
            alpha (float): Degree of prioritization (0 for uniform, 1 for full prioritization).
            max_priority (bool): If use max priority when inserting new experiences
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.alpha = alpha  # Degree of prioritization
        self.device = device
        self.max_priority = max_priority
        self.buffer_size = buffer_size
         # Initialize sum tree
        self.sum_tree = SumTree(buffer_size)
        self.epsilon = 0.2

    def add(self, state, action, reward, next_state, done, td_error=1.0):
        """Add a new experience to memory with the highest priority (for recent experience emphasis)."""
        e = Experience(state, action, reward, next_state, done)
        if self.max_priority:
            priority = max(self.sum_tree.tree[-self.buffer_size:]) if len(self.sum_tree) > 0 else 1.0
        else:
            priority =  self._getPriority(td_error)
        
        self.sum_tree.add(priority, e)

    def sample(self, beta=0.4):
        """
        Sample a batch of experiences with prioritized sampling.

        Args:
            beta (float): Importance-sampling weight for compensation of priority sampling bias.

        Returns:
            tuple: A tuple containing batches of (states, actions, rewards, next_states, dones),
                   sampled indices, and importance-sampling weights.
        """
        experiences = []
        indices = []
        priorities = []
        
        # Segment the total priority to sample each experience proportionally
        segment = self.sum_tree.total_priority() / self.batch_size
        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.sum_tree.get_leaf(s)
            while data is None:
                s = random.uniform(segment * i, segment * (i + 1))
                idx, priority, data = self.sum_tree.get_leaf(s)
            experiences.append(data)
            indices.append(idx)
            priorities.append(priority)
           
        # Normalize sampling probabilities and compute importance-sampling weights
        sampling_probs = np.array(priorities) / self.sum_tree.total_priority()
        weights = (len(self.sum_tree) * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Extract elements from experiences
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities of sampled experiences based on new TD errors.
        
        Args:
            indices (list of int): List of indices corresponding to sampled experiences.
            td_errors (list of float): List of new TD errors for each sampled experience.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._getPriority(td_error)
            self.sum_tree.update(idx, priority)

    def _getPriority(self, td_error):
        return (abs(td_error) + self.epsilon) ** self.alpha


    def __len__(self):
        """Return the current size of the internal memory (number of experiences stored)."""
        return self.sum_tree.size

    def save_buffer(self, file_path):
        """Save the replay buffer to a file using pickle serialization."""
        with open(file_path, 'wb') as f:
            # Use pickle to save the SumTree (including experiences and priorities)
            pickle.dump(self.sum_tree, f)
        print(f"Replay buffer saved to {file_path}")

    def load_buffer(self, file_path):
        """Load the replay buffer from a file."""
        with open(file_path, 'rb') as f:
            # Load the SumTree from the file
            self.sum_tree = pickle.load(f)
        print(f"Replay buffer loaded from {file_path}")


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    Adapted from: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.full(capacity, None, dtype=object)  # Initialize with None for empty slots 
        self.write = 0
        self.size = 0

    def add(self, priority, data):
        """Add a new experience to the tree with given priority."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)  # Update the tree with the new priority
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        self.size = min(self.size + 1, self.capacity)  # Increment size up to capacity

    def update(self, idx, priority):
        """Update priority of the experience at index `idx`."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        self._propogate(idx, change)

    def _propogate(self, idx, change):
        """Propogate the change up the tree to maintain cumulative priorities"""
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propogate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def get_leaf(self, s):
        """Retrieve experience based on cumulative priority `s`."""
        idx = self._retrieve(0, s)

        dataIdx = idx - self.capacity + 1
        if self.data[dataIdx] is None:
            print(f"Warning: Sampled uninitialized data at index {dataIdx}. Returning None.")
            return idx, 0, None  # Return None if data is uninitialized

        return idx, self.tree[idx], self.data[dataIdx]

    def total_priority(self):
        """Return the root node value, representing the total priority sum."""
        return max(self.tree[0], 1e-10)  # Safeguard against zero priority
    
    def __len__(self):
        return self.size