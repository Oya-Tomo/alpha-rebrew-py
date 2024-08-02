import torch
from torch import nn


class PVNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.res_blocks = nn.Sequential(
            ResBlock(3, 128, 128),
            ResBlock(128, 256, 256),
            ResBlock(256, 512, 512),
        )

        self.policy_output = nn.Sequential(
            nn.Conv2d(512, 65, kernel_size=3, padding=1),
            nn.BatchNorm2d(65),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(65 * 8 * 8, 65),
            nn.Softmax(dim=1),
        )

        self.value_output = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res_blocks(x)
        policy = self.policy_output(x.clone())
        value = self.value_output(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.ds = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x = self.ds(x.clone())
        x = self.layers(x)
        return x + skip_x


if __name__ == "__main__":
    from bitboard import Board, Stone

    net = PVNet()

    board = Board()
    print(board.get_actions(Stone.BLACK))
    board.act(Stone.BLACK, 26)

    x = board.to_tensor(Stone.WHITE).reshape(1, 3, 8, 8)
    print(x)

    policy, value = net(x)
    print(policy)
    print(value)
