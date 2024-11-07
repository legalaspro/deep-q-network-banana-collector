import torch.nn as nn
import torch.nn.functional as F

class CNNDuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(CNNDuelingQNetwork, self).__init__()

        # 4 input image channels (grayscale), 32 output channels/feature maps
        # 8x8 square convolution kernel with stride=4
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4) # 20x20x32

        # Second convolutional layer: takes 32 channels and outputs 64 channels/feature maps
        # 4x4 square convolution kernel with stride=2 
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) # 9x9x64
        
        # Third convolutional layer: takes 64 channels and outputs 64 channels/feature maps
        # 3x3 square convolution kernel with stride=1
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1) # 7x7x64
        
        fc_input = 7*7*64
        # Separate fully connected layers for advantage and value streams
        self.fc1_advantage = nn.Linear(fc_input, 256)
        self.fc1_value = nn.Linear(fc_input, 256)

        # Output layers for advantage and value
        self.action_advantage = nn.Linear(256, action_size)
        self.value_state = nn.Linear(256, 1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  

        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 7x7*64)

        # Separate streams for advantage and value
        advantage = F.relu(self.fc1_advantage(x))
        value = F.relu(self.fc1_value(x))

        #Output layer
        action_advantage = self.action_advantage(advantage)
        value_state = self.value_state(value)

        #Q(s,a)=V(s)+(A(s,a)− meanA(s,a))
        action_values = value_state + (action_advantage - action_advantage.mean(dim=1, keepdim=True))
        
        return action_values