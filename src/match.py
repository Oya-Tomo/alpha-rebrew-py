import torch
from torch.multiprocessing import Queue
from bitboard import Board, Stone, flip
from agent import ModelAgent, Step
from mcts import MCT
from models import PVNet


def count_to_score(b: int, w: int, e: int) -> float:
    return (b - w) / (b + w)


def self_play(
    queue: Queue,
    training_weight,
    opponent_weight,
    trainer: Stone,
    sim: int,
    random_begin: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    black_model = PVNet()
    black_model.load_state_dict(
        training_weight if trainer == Stone.BLACK else opponent_weight
    )
    black_model = black_model.to(device)
    black_mct = MCT(black_model, 0.35, 0.1)
    black_agent = ModelAgent(Stone.BLACK, black_mct, sim, 0.01)

    white_model = PVNet()
    white_model.load_state_dict(
        training_weight if trainer == Stone.WHITE else opponent_weight
    )
    white_model = white_model.to(device)
    white_mct = MCT(white_model, 0.35, 0.1)
    white_agent = ModelAgent(Stone.WHITE, white_mct, sim, 0.01)

    board = Board()
    turn = Stone.BLACK

    for _ in range(random_begin):
        if board.is_over():
            break

        if turn == Stone.BLACK:
            action = black_agent.act_random(board)
        else:
            action = white_agent.act_random(board)

        board.act(turn, action)
        turn = flip(turn)

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
        queue.put((black_agent.get_history(), score))
    else:
        queue.put((white_agent.get_history(), -score))


if __name__ == "__main__":
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
                10,
                10,
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
