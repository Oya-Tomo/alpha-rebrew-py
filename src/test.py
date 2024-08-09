import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from agent import ModelAgent
from bitboard import Board, Stone, flip, pos_to_idx
from models import PVNet
from mcts import MCT


def predict_end_result(value, turn) -> float:
    diff = value * 64 * turn
    diff_half = diff / 2
    print(f"black : {32 + diff_half}, white : {32 - diff_half}")
    return diff


def show_value_graph(history):
    k = 2
    avg_history = []
    for i in range(len(history)):
        sample = history[max(0, i - k) : min(len(history), i + k + 1)]
        avg_history.append(sum(sample) / len(sample))

    x = np.array(range(len(avg_history)))
    y = np.array(avg_history)

    fn = make_interp_spline(x, y)

    xs = np.linspace(x.min(), x.max(), 500)
    ys = fn(xs)

    plt.plot(xs, ys)
    plt.show()


def input_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")


def manual_match(tester: Stone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = Board()
    model = PVNet()
    model.load_state_dict(torch.load("checkpoint/model_67.pt")["model"])
    model = model.to(device)

    mct = MCT(model, 100, 0.0001)
    agent = ModelAgent(flip(tester), mct, 1000)

    model.eval()

    turn = Stone.BLACK

    value_history = []

    while not board.is_over():
        print(board)
        actions = board.get_actions(turn)

        _, value = model(board.to_tensor(turn).unsqueeze(0).to(device))
        diff = predict_end_result(value.item(), turn)
        value_history.append(diff)

        if actions == [64]:
            turn = flip(turn)
            continue

        if turn == tester:
            while True:
                x = input_int("x: ")
                y = input_int("y: ")
                action = pos_to_idx(x, y)
                if action in actions:
                    break
                print("Invalid action. Please enter a valid action.")
        else:
            t = time.time()
            action = agent.act(board)
            print(f"Time: {time.time() - t:.2f}s")

        board.act(turn, action)
        turn = flip(turn)

    print(board)

    show_value_graph(value_history)


def random_match(tester: Stone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = Board()
    model = PVNet()
    model.load_state_dict(torch.load("checkpoint/model_21.pt")["model"])
    model = model.to(device)

    mct = MCT(model, 100, 0.01)
    agent = ModelAgent(flip(tester), mct, 1000)

    model.eval()

    turn = Stone.BLACK

    value_history = []

    while not board.is_over():
        print(board)
        actions = board.get_actions(turn)

        _, value = model(board.to_tensor(turn).unsqueeze(0).to(device))
        diff = predict_end_result(value.item(), turn)
        value_history.append(diff)

        if actions == [64]:
            turn = flip(turn)
            continue

        if turn == tester:
            action = random.choice(actions)
        else:
            t = time.time()
            action = agent.act(board)
            print(f"Time: {time.time() - t:.2f}s")

        board.act(turn, action)
        turn = flip(turn)

    print(board)

    show_value_graph(value_history)


if __name__ == "__main__":
    manual_match(Stone.BLACK)
