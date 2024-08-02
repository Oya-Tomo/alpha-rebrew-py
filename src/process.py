from multiprocessing import Process, Queue


class ProcessPool:
    def __init__(self) -> None:
        self.processes: list[Process] = []

    def count(self) -> int:
        return len(self.processes)

    def add(self, process: Process) -> None:
        process.start()
        self.processes.append(process)

    def join_one(self) -> int:
        while True:
            for p_idx in range(len(self.processes)):
                code = self.processes[p_idx].exitcode
                if code != None:
                    self.processes[p_idx].join()
                    self.processes.pop(p_idx)
                    return code
