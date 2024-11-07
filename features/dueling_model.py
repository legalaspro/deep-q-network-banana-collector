import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_size, 128) 
        self.fc2 = nn.Linear(128, 64) 

        # Separate fully connected layers for advantage and value streams
        self.fc3_advantage = nn.Linear(64, 32)
        self.fc3_value = nn.Linear(64, 32)

        # Output layers for advantage and value
        self.action_advantage = nn.Linear(32, action_size)
        self.value_state = nn.Linear(32, 1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Separate streams for advantage and value
        advantage = F.relu(self.fc3_advantage(x))
        value = F.relu(self.fc3_value(x))
        
        #Output layer
        action_advantage = self.action_advantage(advantage)
        value_state = self.value_state(value)

        #Q(s,a)=V(s)+(A(s,a)− meanA(s,a))
        output = value_state + (action_advantage - action_advantage.mean(dim=1, keepdim=True))

        return output
