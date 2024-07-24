import torch
import ray
from bitboard import Board, Stone, flip
from agent import ModelAgent, Step
from mcts import MCT
from models import PVNet


def count_to_score(b: int, w: int, e: int) -> float:
    return (b - w) / (b + w)


@ray.remote
def self_play(training_weight, opponent_weight, trainer: Stone, sim: int) -> tuple[
    list[Step],  # agent move history
    float,  # agent earned score
]:

    black_model = PVNet()
    black_model.load_state_dict(
        training_weight if trainer == Stone.BLACK else opponent_weight
    )
    black_mct = MCT(black_model, 0.35, 0.1)
    black_agent = ModelAgent(Stone.BLACK, black_mct, sim, 0.01)

    white_model = PVNet()
    white_model.load_state_dict(
        training_weight if trainer == Stone.WHITE else opponent_weight
    )
    white_mct = MCT(white_model, 0.35, 0.1)
    white_agent = ModelAgent(Stone.WHITE, white_mct, sim, 0.01)

    board = Board()
    turn = Stone.BLACK

    while not board.is_over():
        if turn == Stone.BLACK:
            action = black_agent.act(board)
        else:
            action = white_agent.act(board)

        board.act(turn, action)
        turn = flip(turn)

    b, w, e = board.get_count()
    score = count_to_score(b, w, e)

    if trainer == Stone.BLACK:
        return (black_agent.get_history(), score)
    else:
        return (white_agent.get_history(), -score)


if __name__ == "__main__":
    ray.init()

    model = PVNet()

    p = self_play.remote(
        model.state_dict(),
        model.state_dict(),
        30,
    )

    result = ray.get(p)
    print(result)
