import re

import inspect

def get_line():
    return inspect.currentframe().f_back.f_lineno

def format_time(sec):
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

class ExpMovingAvg():
    def __init__(self, k):
        self.k = k
        self.avg = 0
        self.num = 0

    def step(self, new_value) -> None:
        self.num += 1
        self.avg = self.avg + (new_value - self.avg) / min(self.num, self.k)

    def get(self) -> float:
        return self.avg

class List():
    def __init__(self, src=None):
        if src is None:
            self.data = []
        elif isinstance(src, list):
            self.data = src
        else:
            self.data = list(src)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in self.data:
            yield i

    def __repr__(self):
        return repr(self.data)

    def ensure(self, *args):
        for arg in args:
            if not arg in self.data:
                self.data.append(arg)

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()