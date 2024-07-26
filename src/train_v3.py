import copy
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import Queue, Process, set_start_method

from bitboard import Stone
from dataloader import PVDataset
from match import self_play
from models import PVNet
from config import Config
from process import ProcessPool

try:
    set_start_method("spawn")
except RuntimeError:
    pass


if __name__ == "__main__":
    cfg = Config(
        warmup_games=3000,
        warmup_mcts_simulation=40,
        num_processes=15,
        loops=10000,
        games=200,
        epochs=20,
        save_epochs=2,
        batch_size=512,
        # optimizer
        lr=0.02,
        weight_decay=0.00001,
        # play
        switch_threshold=0.55,
        mcts_simulation=200,
        random_begin=18,
        value_discount=0.95,
        # history buffer
        data_length=500000,
        data_limit=510000,
        restart_epoch=0,
        load_checkpoint="",
    )

    # Prepare

    print("Prepare started ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.restart_epoch == 0:
        model = PVNet().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        generation = 0

    else:
        checkpoint = torch.load(cfg.load_checkpoint)

        model = PVNet().to(device)
        model.load_state_dict(checkpoint["model"])

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        generation = checkpoint["generation"]

    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    # Self play process
    training_weight = model.cpu().state_dict()
    opponent_weight = model.cpu().state_dict()

    queue = Queue(maxsize=cfg.num_processes)

    self_play_pool = ProcessPool()
    for count in range(cfg.num_processes):
        self_play_pool.add(
            Process(
                target=self_play,
                args=(
                    queue,
                    training_weight,
                    opponent_weight,
                    Stone.BLACK if count % 2 == 0 else Stone.WHITE,
                    cfg.warmup_mcts_simulation,
                    cfg.random_begin,
                ),
            )
        )

    # Warm up for dataset
    print("Warm up started ...")

    dataset = PVDataset(cfg.data_length, cfg.data_limit)

    for count in range(cfg.warmup_games):
        self_play_pool.join_one()
        history, score = queue.get()
        print(f"Warmup game: {count}, {score}")
        dataset.add(
            copy.deepcopy(history),
            copy.deepcopy(score) * cfg.value_discount,
        )
        del history, score

        self_play_pool.add(
            Process(
                target=self_play,
                args=(
                    queue,
                    training_weight,
                    opponent_weight,
                    Stone.BLACK if count % 2 == 0 else Stone.WHITE,
                    cfg.warmup_mcts_simulation,
                    cfg.random_begin,
                ),
            ),
        )

    print("Warm up finished !!")

    # Training loop

    loss_history = []

    for loop in range(cfg.loops):
        lt = time.time()
        print(f"Loop: {loop}")

        training_weight = model.cpu().state_dict()

        win = 0
        lose = 0
        for count in range(cfg.games):
            self_play_pool.join_one()
            history, score = queue.get()
            print(f"    Game: {count}, Score: {score}")
            dataset.add(
                copy.deepcopy(history),
                copy.deepcopy(score) * cfg.value_discount,
            )

            if score > 0:
                win += 1
            elif score < 0:
                lose += 1

            del history, score

            self_play_pool.add(
                Process(
                    target=self_play,
                    args=(
                        queue,
                        training_weight,
                        opponent_weight,
                        Stone.BLACK if count % 2 == 0 else Stone.WHITE,
                        cfg.mcts_simulation,
                        cfg.random_begin,
                    ),
                )
            )

        if win / (win + lose) > cfg.switch_threshold:
            opponent_weight = model.cpu().state_dict()
            generation += 1
            print("    next generation !")

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
        )

        model = model.to(device)

        print(f"    using device : {device}")

        model.train()
        for epoch in range(cfg.epochs):
            et = time.time()
            print(f"    Epoch: {epoch}")

            train_loss = 0.0
            for s, p, v in dataloader:
                s, p, v = s.to(device), p.to(device), v.to(device)

                optimizer.zero_grad()
                p_hat, v_hat = model(s)

                p_loss = policy_criterion(p_hat, p)
                v_loss = value_criterion(v_hat, v)

                loss = p_loss + v_loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"    Loss: {train_loss / len(dataloader)}")
            loss_history.append(train_loss / len(dataloader))

            print(f"    Epoch Time: {time.time() - et}")

        print(f"    Loop Time: {time.time() - lt}")

        if loop % cfg.save_epochs == cfg.save_epochs - 1:
            if not os.path.exists("checkpoint"):
                os.makedirs("checkpoint")
            torch.save(
                {
                    "model": model.cpu().state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generation": generation,
                    "loss_history": loss_history,
                },
                f"checkpoint/model_{loop}.pt",
            )
