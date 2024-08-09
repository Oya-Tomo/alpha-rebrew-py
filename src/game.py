import argparse
import torch
import pygame
from pygame.locals import *

from bitboard import Board, Stone
from mcts import MCT
from models import PVNet

parser = argparse.ArgumentParser(description="Othello AI with AlphaZero")
parser.add_argument("--sim", type=int, default=800, help="number of mcts simulation")
parser.add_argument(
    "-da", "--dirichlet_alpha", type=float, default=10.0, help="dirichlet noise alpha"
)
parser.add_argument(
    "-df", "--dirichlet_frac", type=float, default=0.01, help="dirichlet noise fraction"
)
args = parser.parse_args()


class Predictor:
    def __init__(self):
        # estimator
        checkpoint = torch.load("checkpoint/model_67.pth")
        self.model = PVNet()
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.mct = MCT(self.model, 10.0, 0.01)

    def predict(self, board: Board, color: Stone):
        return self.mct.search(
            board,
            color,
        )


class OthelloGame:
    def __init__(self) -> None:
        # estimator
        checkpoint = torch.load("checkpoint/model_67.pth")
        self.model = PVNet()
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.mct = MCT(self.model, 10.0, 0.01)
