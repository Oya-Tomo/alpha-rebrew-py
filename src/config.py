from dataclasses import dataclass


@dataclass
class GameConfig:
    count: int
    random_start: int  # random moves at the beginning


@dataclass
class SelfPlayConfig:
    num_processes: int
    games: list[GameConfig]
    mcts_num: int


@dataclass
class DatasetConfig:
    periodic_delete: int
    limit_length: int


@dataclass
class TrainConfig:
    loops: int
    epochs: int
    save_epochs: int

    batch_size: int
    lr: float
    weight_decay: float

    restart_epoch: int
    load_checkpoint: str


@dataclass
class Config:
    warmup_config: SelfPlayConfig
    match_config: SelfPlayConfig
    dataset_config: DatasetConfig
    train_config: TrainConfig
