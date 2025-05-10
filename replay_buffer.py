# replay_buffer.py
import random
import numpy as np
from collections import deque # Still useful for the underlying data storage if preferred

# --- SumTree for Efficient Prioritized Sampling ---
# Based on common implementations (e.g., OpenAI Baselines, TF-Agents)
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree structure: Stores priorities. Size is 2*capacity - 1.
        # Leaves are indices capacity-1 to 2*capacity-2.
        self.tree = np.zeros(2 * capacity - 1)
        # Data pointer (circular index for storing actual experiences)
        self.data_pointer = 0
        # Current number of items stored (needed for IS weights)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagates priority changes up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Finds sample index based on cumulative priority value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # Leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Returns the total priority sum (root node)."""
        return self.tree[0]

    def add(self, priority):
        """Adds a new priority to the tree."""
        # Convert external data pointer to internal tree index (leaf node)
        tree_idx = self.data_pointer + self.capacity - 1

        # Update priority in the tree
        self.update(tree_idx, priority)

        # Advance data pointer (circular)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)


    def update(self, tree_idx, priority):
        """Updates the priority of an existing node."""
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
             # This might happen if index from sampling is somehow invalid
             print(f"Warning: Attempted to update invalid tree_idx {tree_idx}")
             return
             
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        """Gets the leaf index, priority, and internal tree index for a sample value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1 # Convert back to the original data pointer index (0 to capacity-1)
        # Handle case where data_idx might point beyond current n_entries if buffer not full?
        # _retrieve should only find leaves corresponding to valid entries if priorities are 0 for unused slots.
        return (idx, self.tree[idx], data_idx)


# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_annealing_steps=1e6, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha # Priority exponent
        self.beta = beta_start # Importance sampling exponent
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon # Small value added to TD error for priority

        self.sum_tree = SumTree(capacity)
        # Store actual experiences - using a list/array is common here
        self.data = [None] * capacity
        # Track max priority for assigning initial priority
        self.max_priority = 1.0
        # Track total steps sampled for beta annealing
        self._total_samples_taken = 0

    def add(self, state, action, reward, next_state, done):
        """Adds an experience with maximum priority initially."""
        experience = (state, action, reward, next_state, done)
        data_idx = self.sum_tree.data_pointer # Index where data will be stored

        self.data[data_idx] = experience # Store data
        # Add to SumTree with current max priority to ensure it gets sampled at least once
        self.sum_tree.add(self.max_priority ** self.alpha)

    def sample(self, batch_size):
        """Samples a batch using priorities and calculates IS weights."""
        experiences = []
        indices = np.empty((batch_size,), dtype=np.int32) # Store tree indices
        data_indices = np.empty((batch_size,), dtype=np.int32) # Store data indices
        weights = np.empty((batch_size,), dtype=np.float32)

        total_p = self.sum_tree.total()
        
        # Add safety check for total priority
        if total_p <= 0:
            print("Warning: Total priority is zero or negative. Using uniform sampling.")
            indices = np.random.choice(min(self.capacity, self.sum_tree.n_entries), batch_size)
            for i, idx in enumerate(indices):
                data_idx = idx
                tree_idx = self.capacity - 1 + data_idx
                weights[i] = 1.0
                indices[i] = tree_idx
                data_indices[i] = data_idx
                experiences.append(self.data[data_idx])
            return (zip(*experiences)), indices, weights
        
        segment = total_p / batch_size

        # Anneal beta
        fraction = min(self._total_samples_taken / self.beta_annealing_steps, 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        self._total_samples_taken += batch_size

        # Max weight for normalization (using current beta)
        # Min probability: min_priority / total_p. Priority is at least (epsilon^alpha)
        # We need n_entries from the tree here.
        min_sampling_prob = ((self.epsilon ** self.alpha) / total_p) if total_p > 0 else 0
        # If buffer is full N = capacity, otherwise N = n_entries
        current_buffer_size = self.sum_tree.n_entries
        
        max_weight = (current_buffer_size * min_sampling_prob) ** (-self.beta) if min_sampling_prob > 0 else 1.0


        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            tree_idx, priority, data_idx = self.sum_tree.get(s)

            if self.data[data_idx] is None:
                 # This can happen if sampling faster than buffer fills initially
                 # or if indices somehow wrap around incorrectly. Resample.
                 # print(f"Warning: Sampled None data at data_idx {data_idx}, tree_idx {tree_idx}. Resampling.")
                 # A simple fix is to resample, a better fix involves ensuring priorities
                 # for empty slots are zero and handling the edge case in get().
                 # For now, let's just try resampling this one slot.
                 s_retry = random.uniform(0, total_p)
                 tree_idx, priority, data_idx = self.sum_tree.get(s_retry)
                 if self.data[data_idx] is None:
                      print(f"CRITICAL WARNING: Resampling failed to find valid data. Buffer state potentially corrupt.")
                      # Fallback: grab a random recent item? Or raise error?
                      # Using the first item as a fallback for now
                      data_idx = 0
                      tree_idx = self.capacity - 1 + data_idx # Corresponding tree index
                      priority = self.sum_tree.tree[tree_idx]


            sampling_probability = priority / total_p
            weights[i] = (current_buffer_size * sampling_probability) ** (-self.beta)

            indices[i] = tree_idx
            data_indices[i] = data_idx
            experiences.append(self.data[data_idx])


        # Normalize weights by dividing by max_weight (to scale down updates)
        weights /= max_weight

        # Unpack experiences for the agent (same structure as before)
        states_dict_tuple, actions_tuple, rewards_tuple, next_states_dict_tuple, dones_tuple = zip(*experiences)

        return (states_dict_tuple, actions_tuple, rewards_tuple, next_states_dict_tuple, dones_tuple), indices, weights

    def update_priorities(self, tree_indices, td_errors):
        """Updates the priorities of sampled experiences based on their TD error."""
        # Add epsilon, raise to alpha
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, priority in zip(tree_indices, priorities):
            # Ensure priority is not zero and not NaN/inf
            priority = max(priority, self.epsilon**self.alpha) # Use epsilon^alpha as minimum
            if np.isnan(priority) or np.isinf(priority):
                 print(f"Warning: NaN or Inf priority calculated ({priority}), using max_priority instead.")
                 priority = self.max_priority ** self.alpha

            self.sum_tree.update(idx, priority)

        # Update max priority seen
        self.max_priority = max(self.max_priority, np.max(priorities**(1/self.alpha))) # Store max TD-error based priority


    def __len__(self):
        return self.sum_tree.n_entries

# --- Standard Replay Buffer (for easy switching via settings) ---
class StandardReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        # Match PER sample output structure for compatibility in agent:
        # Need dummy indices and weights=1.0
        states_dict_tuple, actions_tuple, rewards_tuple, next_states_dict_tuple, dones_tuple = zip(*experiences)
        dummy_indices = np.zeros(batch_size, dtype=np.int32) # Indices not used here
        weights = np.ones(batch_size, dtype=np.float32) # IS weights are 1 for uniform sampling
        return (states_dict_tuple, actions_tuple, rewards_tuple, next_states_dict_tuple, dones_tuple), dummy_indices, weights

    def update_priorities(self, indices, td_errors):
        # Standard buffer doesn't use priorities
        pass

    def __len__(self):
        return len(self.buffer)