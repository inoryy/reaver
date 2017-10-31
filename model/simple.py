import torch
import torch.nn as nn
from model import Model, CNNBlock


class Simple(Model):
    def __init__(self, screen_channels, minimap_channels):
        super().__init__()
        # todo non-spatial features, policy
        self.screen = CNNBlock(*screen_channels)
        self.minimap = CNNBlock(*minimap_channels)
        self.fc1 = nn.Linear((32 + 32) * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 1)
        self.spatial_policy = nn.Conv2d(32 + 32, 1, 1)

    def forward(self, x):
        screen_x, minimap_x = x
        screen = self.screen.forward(screen_x)
        minimap = self.minimap.forward(minimap_x)
        state = torch.cat((screen, minimap), 1)
        spatial_action = self.spatial_policy.forward(state)
        print(screen.size(), minimap.size())
        state = state.view(state.size(0), -1)
        state = nn.ReLU(self.fc1(state))
        value = self.fc2(state)

        return spatial_action, value
