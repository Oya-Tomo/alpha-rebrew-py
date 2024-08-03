import random
import torch
from torch.multiprocessing import Queue
from bitboard import Board, Stone, flip
from agent import ModelAgent, Step
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
    training_weight,
    opponent_weight,
    trainer: Stone,
    mcts_num: int,
    random_start: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    black_model = PVNet()
    black_model.load_state_dict(
        training_weight if trainer == Stone.BLACK else opponent_weight
    )
    black_model = black_model.to(device)
    black_mct = MCT(black_model, 0.9, 0.2)
    black_agent = ModelAgent(Stone.BLACK, black_mct, mcts_num)

    white_model = PVNet()
    white_model.load_state_dict(
        training_weight if trainer == Stone.WHITE else opponent_weight
    )
    white_model = white_model.to(device)
    white_mct = MCT(white_model, 0.9, 0.2)
    white_agent = ModelAgent(Stone.WHITE, white_mct, mcts_num)

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

    if trainer == Stone.BLACK:
        queue.put((black_agent.get_history(), score))
    else:
        queue.put((white_agent.get_history(), -score))


if __name__ == "__main__":
    b, t = generate_random_board(50)
    print(b)

    from multiprocessing import Process

    queue = Queue(maxsize=20)

    training_weight = PVNet().state_dict()
    opponent_weight = PVNet().state_dict()

    p_num = 10

    mp: list[Process] = []
    for i in range(p_num):
        p = Process(
            target=self_play,
            args=(
                queue,
                training_weight,
                opponent_weight,
                Stone.BLACK,
                800,
                40,
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
