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


train_config = Config(
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
        mcts_num=400,
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
        mcts_num=800,
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


debug_config = Config(
    warmup_config=SelfPlayConfig(
        num_processes=15,
        games=[
            GameConfig(count=2, random_start=0),
            GameConfig(count=2, random_start=10),
            GameConfig(count=2, random_start=20),
            GameConfig(count=3, random_start=30),
            GameConfig(count=3, random_start=40),
            GameConfig(count=3, random_start=50),
        ],
        mcts_num=4,
    ),
    match_config=SelfPlayConfig(
        num_processes=15,
        games=[
            GameConfig(count=2, random_start=0),
            GameConfig(count=2, random_start=10),
            GameConfig(count=2, random_start=20),
            GameConfig(count=3, random_start=30),
            GameConfig(count=3, random_start=40),
            GameConfig(count=3, random_start=50),
        ],
        mcts_num=4,
    ),
    dataset_config=DatasetConfig(
        periodic_delete=10,
        limit_length=200,
    ),
    train_config=TrainConfig(
        loops=1000,
        epochs=5,
        save_epochs=100,
        batch_size=512,
        lr=0.005,
        weight_decay=1e-6,
        restart_epoch=0,
        load_checkpoint="",
    ),
)

config = train_config  # export config
