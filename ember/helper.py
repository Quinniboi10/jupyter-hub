import re

import inspect

def get_line() -> int:
    frame = inspect.currentframe()
    return frame.f_back.f_lineno if frame is not None and frame.f_back is not None else -1

def format_time(sec: float) -> str:
    sec = round(sec)

    days = sec // 60 // 60 // 24
    sec %= 60 * 60 * 24

    hours = sec // 60 // 60
    sec %= 60 * 60

    mins = sec // 60
    sec %= 60

    r = ""

    if days > 0:
        r += f"{days:02d}:"
    if hours > 0 or days > 0:
        r += f"{hours:02d}:"
    r += f"{mins:02d}:{sec:02d}"
    return r

def tableify(data: list[list[str]]) -> None:
    cols = list(zip(*data))

    spaces = [len(max(col, key=lambda a: len(a))) for col in cols]

    for row in data:
        for col_idx in range(len(cols)):
            print(f"{row[col_idx]:<{spaces[col_idx]}}", end="  ")
        print()

class ExpMovingAvg():
    def __init__(self, k):
        self.k = k
        self.avg = 0
        self.num = 0

    def step(self, new_value: float) -> None:
        self.num += 1
        self.avg = self.avg + (new_value - self.avg) / min(self.num, self.k)

    def get(self) -> float:
        return self.avg

def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()