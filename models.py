import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        convw = self._conv_output_size(w)
        convh = self._conv_output_size(h)
        linear_input_size = convw * convh * 64
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _conv_output_size(self, size):
        size = (size - 8) // 4 + 1  # First conv layer
        size = (size - 4) // 2 + 1  # Second conv layer
        size = (size - 3) // 1 + 1  # Third conv layer
        return size

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values to [0, 1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
