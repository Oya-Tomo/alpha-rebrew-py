import random
from typing import Generator
import torch
from torch.multiprocessing import Queue, Process
from bitboard import Board, Stone, flip
from agent import ModelAgent, Step
from config import MCTSConfig, SelfPlayConfig
from mcts import MCT, count_to_score
from models import PVNet


def generate_random_board(n: int) -> tuple[Board, Stone]:
    while True:
        _n = n
        board = Board()
        turn = Stone.BLACK

        while _n > 0 and not board.is_over():
            actions = board.get_actions(turn)
            if actions == [64]:
                board.act(turn, 64)
                turn = flip(turn)
                continue

            action = random.choice(actions)
            board.act(turn, action)
            turn = flip(turn)
            _n -= 1

        if not board.is_over():
            return board, turn


def self_play(
    queue: Queue,
    black_weight,
    white_weight,
    config: MCTSConfig,
    random_start: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    black_model = PVNet()
    black_model.load_state_dict(black_weight)
    black_model = black_model.to(device)
    black_mct = MCT(black_model, config)
    black_agent = ModelAgent(Stone.BLACK, black_mct)

    white_model = PVNet()
    white_model.load_state_dict(white_weight)
    white_model = white_model.to(device)
    white_mct = MCT(white_model, config)
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


def self_play_loop(
    config: SelfPlayConfig,
    model: PVNet,
) -> Generator[
    tuple[list[Step], float, list[Step], float], None, None
]:  # yield (steps, score, steps, score)
    queue: Queue = Queue()
    tasks: list[Process] = []
    workers: list[Process] = []

    model_weight = model.cpu().state_dict()

    for _ in range(config.num_self_play):
        tasks.append(
            Process(
                target=self_play,
                args=(
                    queue,
                    model_weight,
                    model_weight,
                    config.mcts_config,
                    random.randint(0, 58),
                ),
            )
        )

    for _ in range(config.num_processes):
        process = tasks.pop(0)
        process.start()
        workers.append(process)

    while len(workers) > 0:
        joined = False
        while True and joined is False:
            for index in range(len(workers)):
                if workers[index].exitcode is not None:
                    workers[index].join()
                    workers.pop(index)
                    joined = True
                    break

        if len(tasks) > 0:
            process = tasks.pop(0)
            process.start()
            workers.append(process)

        black_history, black_score, white_history, white_score = queue.get()
        yield black_history, black_score, white_history, white_score


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
