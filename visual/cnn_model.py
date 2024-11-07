import torch.nn as nn
import torch.nn.functional as F

class CNNQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(CNNQNetwork, self).__init__()

        # 4 input image channels (grayscale), 32 output channels/feature maps
        # 8x8 square convolution kernel with stride=4
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4) # 20x20x32
        # (84+2*0-8)/4 + 1 = 20

        # Second convolutional layer: takes 32 channels and outputs 64 channels/feature maps
        # 4x4 square convolution kernel with stride=2 
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) # 9x9x64
        # (20+2*0-4)/2 + 1 = 9
        
        # Third convolutional layer: takes 64 channels and outputs 64 channels/feature maps
        # 3x3 square convolution kernel with stride=1
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1) # 7x7x64
        # (9+2*0-3)/1 + 1 = 7
        
        #FC Layer 1: 1024 units with ReLU activation
        # Originally 512, but we changed to 1024 to have comparable size network to dueling network
        self.fc1 = nn.Linear(7*7*64, 512) 

        # Output layer for action values
        self.output = nn.Linear(512, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  

        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 7*7*64)

        x = F.relu(self.fc1(x))

        action_values = self.output(x)
        
        return action_values