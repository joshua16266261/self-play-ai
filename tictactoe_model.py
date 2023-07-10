import torch
from torch import nn

class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Flatten(),
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(32, 9)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.torso(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
if __name__ == '__main__':
    traced_script_module = torch.jit.script(TicTacToeNet())
    traced_script_module.save("tictactoe_model.pt")
