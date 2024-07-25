import copy
import os
import time
import torch
from torch.utils.data import DataLoader
import ray

from agent import ModelAgent
from bitboard import Stone
from dataloader import PVDataset
from match import self_play
from models import PVNet
from config import Config


def train():
    ray.init(num_cpus=15, num_gpus=1)

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
        mcts_simulation=100,
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
        origin_model = PVNet()

        training_model = copy.deepcopy(origin_model).to(device)
        training_optimizer = torch.optim.Adam(
            training_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        opponent_model = copy.deepcopy(origin_model).to(device)
        opponent_optimizer = torch.optim.Adam(
            opponent_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    else:
        checkpoint = torch.load(cfg.load_checkpoint)

        training_model = PVNet().to(device)
        training_model.load_state_dict(checkpoint["train"])

        training_optimizer = torch.optim.Adam(
            training_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        training_optimizer.load_state_dict(checkpoint["train_optimizer"])

        opponent_model = PVNet().to(device)
        opponent_model.load_state_dict(checkpoint["opponent"])
        opponent_optimizer = torch.optim.Adam(
            opponent_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        opponent_optimizer.load_state_dict(checkpoint["opponent_optimizer"])

    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    # Self play process
    training_weight = ray.put(training_model.cpu().state_dict())
    opponent_weight = ray.put(opponent_model.cpu().state_dict())

    working_self_play: list = [
        self_play.remote(
            training_weight,
            opponent_weight,
            Stone.BLACK if count % 2 == 0 else Stone.WHITE,
            cfg.warmup_mcts_simulation,
        )
        for count in range(cfg.games)
    ]

    # Warm up for dataset
    print("Warm up started ...")

    dataset = PVDataset(cfg.data_length, cfg.data_limit)

    for count in range(cfg.warmup_games):
        fin, working_self_play = ray.wait(working_self_play, num_returns=1)
        history, score = ray.get(fin[0])
        dataset.add(history, score * cfg.value_discount)
        print(f"    Game: {count}, Score: {score}")

        working_self_play.append(
            self_play.remote(
                training_weight,
                opponent_weight,
                Stone.BLACK if count % 2 == 0 else Stone.WHITE,
                cfg.warmup_mcts_simulation,
            )
        )

    print("Warm up finished !!")

    # Train loop

    loss_history = []

    for loop in range(cfg.loops):
        lt = time.time()
        print(f"Loop: {loop}")

        training_weight = ray.put(training_model.cpu().state_dict())
        opponent_weight = ray.put(opponent_model.cpu().state_dict())

        win = 0
        lose = 0
        for count in range(cfg.games):
            fin, working_self_play = ray.wait(working_self_play, num_returns=1)
            history, score = ray.get(fin[0])
            dataset.add(history, score * cfg.value_discount)
            print(f"    Game: {count}, Score: {score}")

            if score > 0:
                win += 1
            elif score < 0:
                lose += 1

            working_self_play.append(
                self_play.remote(
                    training_weight,
                    opponent_weight,
                    Stone.BLACK if count % 2 == 0 else Stone.WHITE,
                    cfg.mcts_simulation,
                )
            )

        if win / (win + lose) > cfg.switch_threshold:
            training_model, opponent_model = (
                opponent_model,
                training_model,
            )
            training_optimizer, opponent_optimizer = (
                opponent_optimizer,
                training_optimizer,
            )
            print("    switched trainer !")

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
        )

        training_model = training_model.to(device)

        print(f"    using device : {device}")

        training_model.train()
        for epoch in range(cfg.epochs):
            et = time.time()
            print(f"    Epoch: {epoch}")

            train_loss = 0.0
            for s, p, v in dataloader:
                s, p, v = s.to(device), p.to(device), v.to(device)

                training_optimizer.zero_grad()
                p_hat, v_hat = training_model(s)

                p_loss = policy_criterion(p_hat, p)
                v_loss = value_criterion(v_hat, v)

                loss = p_loss + v_loss

                loss.backward()
                training_optimizer.step()

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
                    "train": training_model.cpu().state_dict(),
                    "train_optimizer": training_optimizer.state_dict(),
                    "opponent": opponent_model.cpu().state_dict(),
                    "opponent_optimizer": opponent_optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                f"checkpoint/model_{loop}.pt",
            )


if __name__ == "__main__":
    train()
