import copy
import os
import pprint
import sys
import time
from typing import Generator
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import Queue, Process, set_start_method
from tqdm import tqdm

from agent import Step
from bitboard import Stone
from dataloader import PVDataset
from match import self_play
from models import PVNet
from config import (
    SelfPlayConfig,
    config,
)


def self_play_loop(
    config: SelfPlayConfig,
    model: PVNet,
    queue: Queue,
) -> Generator[
    tuple[list[Step], float, list[Step], float], None, None
]:  # yield (steps, score, steps, score)
    tasks: list[Process] = []
    workers: list[Process] = []

    model_weight = model.cpu().state_dict()

    for game in config.game_config:
        for i in range(game.count):
            task = Process(
                target=self_play,
                args=(
                    queue,
                    model_weight,
                    model_weight,
                    config.mcts_config,
                    game.random_start,
                ),
            )
            tasks.append(task)

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


def train():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    queue = Queue(maxsize=20)
    dataset = PVDataset(config.dataset_config.limit_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.train_config.restart_epoch == 0:
        model = PVNet().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train_config.lr,
            weight_decay=config.train_config.weight_decay,
        )
    else:
        checkpoint = torch.load(config.train_config.load_checkpoint)
        model = PVNet().to(device)

        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train_config.lr,
            weight_decay=config.train_config.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("Warmup Start")

    if config.train_config.load_dataset is not None:
        print("Load Dataset")
        dataset.load_state_dict(torch.load(config.train_config.load_dataset))
        print(f"Dataset Length: {len(dataset)}")
    else:
        for i, res in enumerate(self_play_loop(config.warmup_config, model, queue)):
            black_history, black_score, white_history, white_score = res
            dataset.add(black_history, black_score)
            dataset.add(white_history, white_score)
            print(f"    Game {i} Score: {black_score}")

    print("Warmup Finish")

    if config.train_config.save_dataset is not None:
        print("Save Dataset")
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")
        torch.save(dataset.state_dict(), config.train_config.save_dataset)

    loss_history = []

    for loop in range(config.train_config.loops):
        print(f"Loop {loop} Start")

        dataset.periodic_delete(config.dataset_config.periodic_delete)
        for i, res in enumerate(self_play_loop(config.match_config, model, queue)):
            black_history, black_score, white_history, white_score = res
            dataset.add(black_history, black_score)
            dataset.add(white_history, white_score)
            print(f"    Game {i} Score: {black_score}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.train_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model = model.to(device)
        model.train()

        with tqdm(
            range(config.train_config.epochs),
            bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
        ) as pbar:
            epoch_loss_history = []
            for epoch in pbar:
                total_loss = 0
                pbar.set_description(f"Epoch {epoch}")

                for state, policy, value in tqdm(
                    dataloader,
                    bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
                    leave=False,
                ):
                    state = state.to(device)
                    policy = policy.to(device)
                    value = value.to(device)

                    optimizer.zero_grad()

                    policy_pred, value_pred = model(state)
                    loss_policy = torch.nn.functional.cross_entropy(policy_pred, policy)
                    loss_value = torch.nn.functional.mse_loss(value_pred, value)
                    loss = loss_policy + loss_value

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                epoch_loss_history.append(total_loss / len(dataloader))
                loss_history.append(total_loss / len(dataloader))
                pbar.set_postfix({"loss": total_loss / len(dataloader)})

            print("loss in epoch")
            pprint.pprint(epoch_loss_history)

        if (
            loop % config.train_config.save_epochs
            == config.train_config.save_epochs - 1
        ):
            if not os.path.exists("checkpoint"):
                os.makedirs("checkpoint")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                f"checkpoint/model_{loop}.pt",
            )


if __name__ == "__main__":
    train()
