import copy
import random
from dataclasses import dataclass
import numpy as np

from bitboard import Board, Stone
from mcts import MCT


@dataclass
class Step:
    state: Board
    turn: Stone
    policy: float


class ModelAgent:
    # ModelAgent
    #     MCTS + e-greedy decision process
    #     buffer: [(state, turn, policy)]

    def __init__(self, stone: Stone, mct: MCT, mcts_num: int) -> None:
        self.stone = stone
        self.mct = mct
        self.mcts_num = mcts_num

        self.buffer: list[Step] = []  # (state, policy)

    def act(self, board: Board) -> int:
        actions = board.get_actions(self.stone)

        policy = self.mct.search(board, self.stone, self.mcts_num)
        self.buffer.append(
            Step(state=copy.deepcopy(board), turn=self.stone, policy=policy)
        )
        policy = np.array(policy, dtype=np.float32)

        while True:
            action = int(random.choice(np.where(policy == policy.max())[0]))

            if action in actions:
                break
            policy[action] = 0

        return action

    # will be deleted
    def act_random(self, board: Board) -> int:
        actions = board.get_actions(self.stone)

        policy = self.mct.search(board, self.stone, self.mcts_num)
        self.buffer.append(
            Step(state=copy.deepcopy(board), turn=self.stone, policy=policy)
        )
        action = random.choice(actions)
        return action

    def get_history(self):
        return self.buffer
