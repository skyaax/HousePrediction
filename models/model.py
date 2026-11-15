import torch
import torch.nn as nn
class Wide_and_Deep(nn.Module):
    def __init__(self, wide_X, deep_X):
        super().__init__()
        self.deep = nn.Sequential(
            nn.Linear(deep_X,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.output = nn.Linear(wide_X+32, 1)


    def forward(self,wide,deep):
        deep_output = self.deep(deep)
        combined = torch.cat(
            [wide,deep_output],
            dim =1
        )
        return self.output(combined)