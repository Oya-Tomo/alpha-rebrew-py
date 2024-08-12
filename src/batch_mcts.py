import math
import random
import copy
from dataclasses import dataclass
from multiprocessing.connection import Connection
import torch
import numpy as np

from bitboard import ACTION_COUNT, Board, Stone, flip
from config import MCTSConfig
from models import PVNet


def sigmoid(x: float, a: float):
    return 1 / (1 + math.exp(-a * x))


def logit(x: float, a: float):
    return math.log(x / (1 - x)) / a


# black win -> positive +
# white win -> negative -
# draw      -> 0
def count_to_score(b: int, w: int) -> float:
    count_score = (b - w) / (b + w)
    return sigmoid(count_score, 6) * 2 - 1


def score_to_count(score: float) -> tuple[int, int]:
    score_sigm = (score + 1) / 2
    count_score = logit(score_sigm, 6)
    diff = count_score * 64
    b = 32 + diff / 2
    w = 32 - diff / 2
    return b, w


@dataclass
class BatchInputItem:
    state: Board
    turn: Stone


@dataclass
class BatchOutputItem:
    policy: list[float]
    value: float


class BatchMCT:
    def __init__(
        self,
        pipe: Connection,
        config: MCTSConfig,
    ) -> None:
        self.pipe = pipe
        self.config = config

        self.P: dict[int, list[float]] = {}  # policy
        self.V: dict[int, float] = {}  # value

        self.N: dict[int, list[int]] = {}  # action visit count
        self.S: dict[int, list[float]] = {}  # total earned score

        self.transition_cache: dict[int, list[Board]] = {}

    def C_puct(self, s: int):
        return self.config.c_init  # not much influence
        # return (
        #     math.log((1 + self.N_s(s) + self.config.c_base) / self.config.c_base)
        #     + self.config.c_init
        # )

    def N_s(self, s: int) -> int | None:
        if s in self.N:
            return sum(self.N[s])
        else:
            return None

    def N_sa(self, s: int, a: int) -> int | None:
        if s in self.N:
            return self.N[s][a]
        else:
            return None

    def search(self, state: Board, turn: Stone):
        s = state.to_key(turn)

        if self.N_s(s) == None:
            self.expand(state, turn)

        actions = state.get_actions(turn)

        dirichlet_noise = np.random.dirichlet(
            alpha=[self.config.dirichlet_alpha] * len(actions)
        )
        for action, noise in zip(actions, dirichlet_noise):
            self.P[s][action] = (
                self.config.dirichlet_frac * noise
                + (1 - self.config.dirichlet_frac) * self.P[s][action]
            )

        for _ in range(self.config.simulation):
            U = [
                self.C_puct(s)
                * self.P[s][a]
                * math.sqrt(self.N_s(s))
                / (1 + self.N_sa(s, a))
                for a in range(ACTION_COUNT)
            ]

            Q = [q / n if n != 0 else q for q, n in zip(self.S[s], self.N[s])]

            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array(
                [
                    score if action in actions else -np.inf
                    for action, score in enumerate(scores)
                ]
            )
            action = random.choice(np.where(scores == scores.max())[0])
            next_state = self.transition_cache[s][action]

            value = -self.evaluate(next_state, flip(turn))

            self.S[s][action] += value
            self.N[s][action] += 1

        mcts_policy = [n / self.N_s(s) for n in self.N[s]]
        return mcts_policy

    def expand(self, state: Board, turn: Stone):
        s = state.to_key(turn)

        self.N[s] = [0] * ACTION_COUNT
        self.S[s] = [0] * ACTION_COUNT
        self.transition_cache[s] = [None] * ACTION_COUNT

        actions = state.get_actions(turn)
        for action in actions:
            next_state = copy.deepcopy(state)
            next_state.act(turn, action)
            self.transition_cache[s][action] = next_state

        self.pipe.send(
            BatchInputItem(
                state=state,
                turn=turn,
            )
        )
        output: BatchOutputItem = self.pipe.recv()

        self.P[s] = output.policy
        self.V[s] = output.value

        return self.V[s]

    def evaluate(self, state: Board, turn: Stone) -> float:
        s = state.to_key(turn)

        if state.is_over():
            b, w, e = state.get_count()
            score = count_to_score(b, w)
            if turn == Stone.BLACK:
                return score
            else:
                return -score
        elif self.N_s(s) == None:
            value = self.expand(state, turn)
            return value
        else:
            U = [
                self.C_puct(s)
                * self.P[s][a]
                * math.sqrt(self.N_s(s))
                / (1 + self.N_sa(s, a))
                for a in range(ACTION_COUNT)
            ]

            Q = [q / n if n != 0 else q for q, n in zip(self.S[s], self.N[s])]

            actions = state.get_actions(turn)

            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array(
                [
                    score if action in actions else -np.inf
                    for action, score in enumerate(scores)
                ]
            )
            action = random.choice(np.where(scores == scores.max())[0])
            next_state = self.transition_cache[s][action]

            value = -self.evaluate(next_state, flip(turn))

            self.S[s][action] += value
            self.N[s][action] += 1

            return value


def predict_process(weight, pipes: list[Connection], batch: int, recv_loop: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PVNet()
    model.load_state_dict(weight)
    model = model.to(device)
    model.eval()

    while True:
        ids = []
        inputs = []
        for _ in range(recv_loop):
            for id, pipe in enumerate(pipes):
                if len(inputs) >= batch:
                    break
                if pipe.poll():
                    data: BatchInputItem = pipe.recv()
                    ids.append(id)
                    inputs.append(data.state.to_tensor(data.turn))

        if len(inputs) == 0:
            continue

        inputs = torch.stack(inputs).to(device)
        with torch.no_grad():
            policies, values = model(inputs)
            policies: torch.Tensor = policies.cpu().reshape(-1, ACTION_COUNT)
            values: torch.Tensor = values.cpu().flatten()

        for i, id in enumerate(ids):
            pipes[id].send(
                BatchOutputItem(
                    policy=policies[i].tolist(),
                    value=values[i].item(),
                )
            )


if __name__ == "__main__":
    b = 40
    w = 24
    score = count_to_score(b, w)
    print(score)
    b, w = score_to_count(score)
    print(b, w)
