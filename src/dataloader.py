import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from agent import Step


class PVDataset:
    def __init__(self, limit: int) -> None:
        self.buffer: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        self.limit = limit

    def add(self, history: list[Step], score: float):
        for step in history:
            self.buffer.append(
                (
                    step.state.to_tensor(step.turn),
                    torch.tensor(step.policy),
                    torch.tensor([score]),
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
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return {
            "buffer": self.buffer,
        }

    def load_state_dict(self, state_dict):
        self.buffer = state_dict["buffer"]
        return self
