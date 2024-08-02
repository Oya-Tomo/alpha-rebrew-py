import copy
import os
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
from config import Config, DatasetConfig, GameConfig, SelfPlayConfig, TrainConfig


def self_play_loop(
    config: SelfPlayConfig,
    model: PVNet,
    queue: Queue,
) -> Generator[tuple[list[Step], float]]:  # yield (steps, score)
    tasks: list[Process] = []
    workers: list[Process] = []

    model_weight = model.cpu().state_dict()

    for game in config.games:
        for i in range(game.count):
            stone = Stone.BLACK if i % 2 == 0 else Stone.WHITE
            task = Process(
                target=self_play,
                args=(queue, model_weight, model_weight, stone, game.random_start),
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

        history, score = queue.get()
        yield history, score


def train():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    config = Config(
        warmup_config=SelfPlayConfig(
            num_processes=15,
            games=[
                GameConfig(count=500, random_start=0),
                GameConfig(count=500, random_start=10),
                GameConfig(count=500, random_start=20),
                GameConfig(count=1000, random_start=30),
                GameConfig(count=1000, random_start=40),
                GameConfig(count=1000, random_start=50),
            ],
        ),
        match_config=SelfPlayConfig(
            num_processes=15,
            games=[
                GameConfig(count=50, random_start=0),
                GameConfig(count=50, random_start=10),
                GameConfig(count=50, random_start=20),
                GameConfig(count=100, random_start=30),
                GameConfig(count=100, random_start=40),
                GameConfig(count=100, random_start=50),
            ],
        ),
        dataset_config=DatasetConfig(
            periodic_delete=2000,
            limit_length=500000,
        ),
        train_config=TrainConfig(
            loops=1000,
            epochs=50,
            save_epochs=2,
            batch_size=512,
            lr=0.005,
            weight_decay=1e-6,
            restart_epoch=0,
            load_checkpoint="",
        ),
    )

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

    for history, score in self_play_loop(config.warmup_config, model, queue):
        dataset.add(history, score)

    print("Warmup Finish")

    loss_history = []

    for loop in range(config.train_config.loops):
        print(f"Loop {loop} Start")

        dataset.periodic_delete(config.dataset_config.periodic_delete)
        for history, score in self_play_loop(config.match_config, model, queue):
            dataset.add(history, score)
            print(f"    Score: {score}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.train_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model = model.to(device)
        model.train()

        with tqdm(range(config.train_config.epochs)) as pbar:
            for epoch in pbar:
                total_loss = 0
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix({"loss": total_loss / len(dataloader)})

                for state, policy, value in tqdm(dataloader):
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
                loss_history.append(total_loss / len(dataloader))

        if (
            loop % config.train_config.save_epochs
            == config.train_config.save_epochs - 1
        ):
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
