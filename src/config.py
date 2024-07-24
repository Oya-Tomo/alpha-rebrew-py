from dataclasses import dataclass


@dataclass
class Config:
    loops: int
    games: int
    epochs: int
    save_epochs: int  # spoch / save

    batch_size: int
    lr: float
    weight_decay: float

    switch_threshold: float  # win rate
    mcts_simulation: int  # mcts simulation count
    value_discount: float  # value discount rate

    data_length: int
    data_limit: int

    restart_epoch: int  # 0: start from 0
    load_checkpoint: str  # if restart_epoch > 0, it will be loaded.
