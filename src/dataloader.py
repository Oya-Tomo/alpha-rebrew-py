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
    def __init__(self, limit: int) -> None:
        self.buffer: list[PVItem] = []
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
            p = len(self.buffer) - self.limit
            self.buffer = self.buffer[p:]

    def periodic_delete(self, n: int):
        if len(self.buffer) > n:
            self.buffer = self.buffer[n:]
        else:
            assert False, "buffer length will be 0 !!"

    def __getitem__(self, index: int):
        item = self.buffer[index]
        return (
            item.state,
            item.policy,
            item.value,
        )

    def __len__(self):
        return len(self.buffer)
