import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from agent import Step


@dataclass
class PVItem:
    state: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


class PVDataset:
    def __init__(self, length: int, limit: int) -> None:
        assert length < limit, "length must be less than limit."

        self.buffer: list[PVItem] = []

        self.length = length
        self.limit = limit

    def add(self, history: list[Step], score: float):
        for step in history:
            self.buffer.append(
                PVItem(
                    state=step.state.to_tensor(step.turn),
                    policy=torch.tensor(step.policy),
                    value=torch.tensor([score]),
                ),
            )

        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[self.length :]

    def enough_data(self):
        return len(self.buffer) >= self.length * 0.95

    def __getitem__(self, index: int):
        item = self.buffer[index]
        return (
            item.state,
            item.policy,
            item.value,
        )

    def __len__(self):
        return len(self.buffer)
