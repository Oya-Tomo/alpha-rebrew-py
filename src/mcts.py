import copy
import random
import torch
import numpy as np
import math

from bitboard import ACTION_COUNT, Board, Stone, flip
from config import MCTSConfig


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


class MCT:
    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device

        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_frac = config.dirichlet_frac

        self.c_base = config.c_base
        self.c_init = config.c_init
        # ref: https://tadaoyamaoka.hatenablog.com/entry/2018/12/08/191619

        self.P: dict[int, list[float]] = {}  # prior probability

        self.N: dict[int, list[int]] = {}  # action visit count
        self.S: dict[int, list[float]] = {}  # total earned score

        self.transition_cache = {}

    def _c_puct(self, s: int):
        return math.log((1 + sum(self.N[s]) + self.c_base) / self.c_base) + self.c_init

    def search(self, state: Board, turn: Stone, sim: int):
        s = state.to_key(turn)

        if not (s in self.P):
            _ = self.expand(state, turn)

        actions = state.get_actions(turn)

        dirichlet_noise = np.random.dirichlet(
            alpha=[self.dirichlet_alpha] * len(actions)
        )
        for a, noise in zip(actions, dirichlet_noise):
            self.P[s][a] = (
                self.dirichlet_frac * noise + (1 - self.dirichlet_frac) * self.P[s][a]
            )

        for _ in range(sim):
            U = [
                self._c_puct(s)
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
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

        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]
        return mcts_policy

    def expand(self, state: Board, turn: Stone):
        s = state.to_key(turn)

        with torch.no_grad():
            inputs = state.to_tensor(turn).to(self.device).reshape([1, 3, 8, 8])
            outputs: torch.Tensor = self.model(inputs)
            policy: torch.Tensor = outputs[0].reshape([ACTION_COUNT])
            value: torch.Tensor = outputs[1].item()

        self.P[s] = policy.tolist()
        self.N[s] = [0] * ACTION_COUNT
        self.S[s] = [0] * ACTION_COUNT
        self.transition_cache[s] = [0] * ACTION_COUNT

        actions = state.get_actions(turn)

        for action in range(ACTION_COUNT):
            if action in actions:
                next_state = copy.deepcopy(state)
                next_state.act(turn, action)
                self.transition_cache[s][action] = next_state
            else:
                self.transition_cache[s][action] = None

        return value

    def evaluate(self, state: Board, turn: Stone) -> float:
        s = state.to_key(turn)

        if state.is_over():
            b, w, e = state.get_count()
            score = count_to_score(b, w)
            if turn == Stone.BLACK:
                return score
            else:
                return -score
        elif not (s in self.P):
            value = self.expand(state, turn)
            return value
        else:
            U = [
                self._c_puct(s)
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
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


if __name__ == "__main__":
    b = 40
    w = 24
    score = count_to_score(b, w)
    print(score)
    b, w = score_to_count(score)
    print(b, w)
