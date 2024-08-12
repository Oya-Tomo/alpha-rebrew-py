from multiprocessing.connection import Connection
import random
import torch
from torch.multiprocessing import Queue
from batch_mcts import BatchMCT
from bitboard import Board, Stone, flip
from agent import ModelAgent
from config import MCTSConfig
from mcts import MCT, count_to_score
from models import PVNet


def generate_random_board(random_start: int) -> tuple[Board, Stone]:
    while True:
        board = Board()
        turn = Stone.BLACK

        for _ in range(random_start):
            if board.is_over():
                break
            actions = board.get_actions(turn)
            action = random.choice(actions)
            board.act(turn, action)
            turn = flip(turn)

        if not board.is_over():
            break

    return board, turn


def self_play(
    queue: Queue,
    pipe: Connection,
    config: MCTSConfig,
    random_start: int,
) -> None:
    black_mct = BatchMCT(pipe, config)
    black_agent = ModelAgent(Stone.BLACK, black_mct)

    white_mct = BatchMCT(pipe, config)
    white_agent = ModelAgent(Stone.WHITE, white_mct)

    board, turn = generate_random_board(random_start)

    while not board.is_over():
        if turn == Stone.BLACK:
            action = black_agent.act(board)
        else:
            action = white_agent.act(board)

        board.act(turn, action)
        turn = flip(turn)

    b, w, e = board.get_count()
    score = count_to_score(b, w)

    queue.put((black_agent.get_history(), score, white_agent.get_history(), -score))


if __name__ == "__main__":
    b, t = generate_random_board(50)
    print(b)

    from multiprocessing import Process

    queue = Queue(maxsize=20)

    black_weight = PVNet().state_dict()
    white_weight = PVNet().state_dict()

    config = MCTSConfig(
        simulation=10,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        c_base=19652,
        c_init=1.25,
    )

    p_num = 10

    mp: list[Process] = []
    for i in range(p_num):
        p = Process(
            target=self_play,
            args=(
                queue,
                black_weight,
                white_weight,
                config,
                30,
            ),
        )
        p.start()
        mp.append(p)

    for i in range(p_num):
        found = False
        while not found:
            for pidx in range(len(mp)):
                if mp[pidx].exitcode != None:
                    mp[pidx].join()
                    mp.pop(pidx)
                    found = True
                    break

        result = queue.get()
        print(i)

        print(result)
        del result
