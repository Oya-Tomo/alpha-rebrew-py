from dataclasses import dataclass


@dataclass
class MCTSConfig:
    simulation: int
    dirichlet_alpha: float
    dirichlet_frac: float
    c_init: float
    c_base: float


@dataclass
class GameConfig:
    count: int
    random_start: int  # random moves at the beginning


@dataclass
class SelfPlayConfig:
    num_processes: int
    game_config: list[GameConfig]
    mcts_config: MCTSConfig


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
    step_size: int
    gamma: float

    restart_epoch: int
    load_checkpoint: str

    save_dataset: str | None
    load_dataset: str | None


@dataclass
class Config:
    warmup_config: SelfPlayConfig
    match_config: SelfPlayConfig
    dataset_config: DatasetConfig
    train_config: TrainConfig


train_config = Config(
    warmup_config=SelfPlayConfig(
        num_processes=13,
        game_config=[
            GameConfig(count=200, random_start=0),
            GameConfig(count=200, random_start=5),
            GameConfig(count=200, random_start=10),
            GameConfig(count=200, random_start=15),
            GameConfig(count=200, random_start=20),
            GameConfig(count=500, random_start=25),
            GameConfig(count=500, random_start=30),
            GameConfig(count=500, random_start=35),
            GameConfig(count=500, random_start=40),
            GameConfig(count=1000, random_start=45),
            GameConfig(count=1000, random_start=50),
            GameConfig(count=1000, random_start=55),
        ],
        mcts_config=MCTSConfig(
            simulation=800,
            dirichlet_alpha=0.9,
            dirichlet_frac=0.2,
            c_init=1.0,
            c_base=19652,
        ),
    ),
    match_config=SelfPlayConfig(
        num_processes=13,
        game_config=[
            GameConfig(count=20, random_start=0),
            GameConfig(count=20, random_start=5),
            GameConfig(count=20, random_start=10),
            GameConfig(count=20, random_start=15),
            GameConfig(count=20, random_start=20),
            GameConfig(count=50, random_start=25),
            GameConfig(count=50, random_start=30),
            GameConfig(count=50, random_start=35),
            GameConfig(count=50, random_start=40),
            GameConfig(count=100, random_start=45),
            GameConfig(count=100, random_start=50),
            GameConfig(count=100, random_start=55),
        ],
        mcts_config=MCTSConfig(
            simulation=800,
            dirichlet_alpha=0.9,
            dirichlet_frac=0.2,
            c_init=1.0,
            c_base=19652,
        ),
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
        step_size=10,
        gamma=0.5,
        restart_epoch=0,
        load_checkpoint="",
        save_dataset="checkpoint/dataset.pt",
        load_dataset=None,
    ),
)


debug_config = Config(
    warmup_config=SelfPlayConfig(
        num_processes=12,
        game_config=[
            GameConfig(count=2, random_start=0),
            GameConfig(count=2, random_start=10),
            GameConfig(count=2, random_start=20),
            GameConfig(count=3, random_start=30),
            GameConfig(count=3, random_start=40),
            GameConfig(count=3, random_start=50),
        ],
        mcts_config=MCTSConfig(
            simulation=10,
            dirichlet_alpha=0.9,
            dirichlet_frac=0.2,
            c_init=1.0,
            c_base=19652,
        ),
    ),
    match_config=SelfPlayConfig(
        num_processes=15,
        game_config=[
            GameConfig(count=2, random_start=0),
            GameConfig(count=2, random_start=10),
            GameConfig(count=2, random_start=20),
            GameConfig(count=3, random_start=30),
            GameConfig(count=3, random_start=40),
            GameConfig(count=3, random_start=50),
        ],
        mcts_config=MCTSConfig(
            simulation=10,
            dirichlet_alpha=0.9,
            dirichlet_frac=0.2,
            c_init=1.0,
            c_base=19652,
        ),
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
        step_size=10,
        gamma=0.5,
        restart_epoch=0,
        load_checkpoint="",
        save_dataset=None,
        load_dataset=None,
    ),
)

config = train_config  # export config
