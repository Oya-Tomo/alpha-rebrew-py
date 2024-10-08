import os
import pprint
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from tqdm import tqdm

from dataloader import PVDataset
from match import self_play_loop
from models import PVNet
from config import config


def train():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    dataset = PVDataset(config.dataset_config.limit_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.train_config.restart_epoch == 0:
        model = PVNet().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train_config.lr,
            weight_decay=config.train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train_config.step_size,
            gamma=config.train_config.gamma,
        )
        loss_history = []
    else:
        checkpoint = torch.load(config.train_config.load_checkpoint)
        model = PVNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train_config.lr,
            weight_decay=config.train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train_config.step_size,
            gamma=config.train_config.gamma,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        loss_history = checkpoint["loss_history"]

    print("Warmup Start")

    if config.train_config.load_dataset is not None:
        print("Load Dataset")
        dataset.load_state_dict(torch.load(config.train_config.load_dataset))
        print(f"Dataset Length: {len(dataset)}")
    else:
        for i, res in enumerate(self_play_loop(config.warmup_config, model)):
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

    for loop in range(config.train_config.loops):
        print(f"Loop {loop} Start")

        dataset.periodic_delete(config.dataset_config.periodic_delete)
        for i, res in enumerate(self_play_loop(config.match_config, model)):
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
            ncols=80,
            bar_format="{l_bar}{bar:10}{r_bar}",
        ) as pbar:
            epoch_loss_history = []
            for epoch in pbar:
                total_loss = 0

                for state, policy, value in tqdm(
                    dataloader,
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    leave=False,
                    ncols=60,
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

        print(f"Scheduled Learning Rate: {scheduler.get_last_lr()}")
        scheduler.step()

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
                    "scheduler": scheduler.state_dict(),
                    "loss_history": loss_history,
                },
                f"checkpoint/model_{loop}.pt",
            )
        if config.train_config.save_dataset is not None:
            if not os.path.exists("checkpoint"):
                os.makedirs("checkpoint")
            print(f"Save Dataset : len {len(dataset)}")
            torch.save(dataset.state_dict(), config.train_config.save_dataset)


if __name__ == "__main__":
    train()
