import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import warnings
import math

from .helper import *
from .metric import *

'''
Valid callback positions

BEFORE_FIT
BEFORE_TRAIN
BEFORE_VALID
BEFORE_BATCH

AFTER_FIT
AFTER_TRAIN
AFTER_VALID
AFTER_BATCH
'''

# Overarching class from which all other
# callbacks must inheret
class Callback():
    def __init__(self):
        self.learner = None

    def __repr__(self):
        return type(self).__name__

    def set_learner(self, learner):
        self.learner = learner

    @property
    def name(self):
        return camel_to_snake(self.__repr__())

# Toggles a model between train
# and eval mode. This callback was
# more of a proof-of-concept, not
# a neccesity
class TrainEvalCallback(Callback):
    def before_train(self):
        self.learner.model.train()
    def before_valid(self):
        self.learner.model.eval()

# Records useful information about the
# learner as it runs. Some metrics are
# required for correct function of other
# compoents and are always enabled such
# as LR and loss

# Snapshot wraps:
#    self.time: float    - time of the snapshot since start of training
#    self.step: int      - number of steps recorded at time of snapshot
#    self.metrics: dict  - (metric, value) pairs
#    self.is_train: bool - was this snapshot taken during train or valid
class RecorderCallback(Callback):
    @dataclass
    class Snapshot():
        elapsed: float
        step: int
        metrics: dict
        is_train: bool

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

        self.step = 0

        self.data = []

        self.start = None

    def set_learner(self, learner):
        self.learner = learner
        for m in self.metrics:
            m.set_learner(learner)

    @property
    def name(self):
        return "recorder"

    def plot(self, *metric_names, y_label=None):
        x = []
        y = []

        for m_name in metric_names:
            m_x = []
            m_y = []
            for i in range(0, len(self.data)):
                s = self.data[i]
                if m_name in s.metrics:
                    m_x.append(s.step)
                    m_y.append(s.metrics[m_name])
            x.append(m_x)
            y.append(m_y)

        for m_x, m_y in zip(x, y):
            plt.plot(m_x, m_y)

        if y_label is None:
            y_label = ''.join(f"{name}, " for name in metric_names)[:-2]
        plt.xlabel("Batch")
        plt.ylabel(y_label)
        plt.grid(True)

        plt.show()

    def _collect_metrics(self, is_train):
        metrics = {}

        for m in self.metrics:
            if m.is_train == is_train:
                value = m.value
                if value is not None:
                    metrics[m.name] = value

        self.data.append(self.Snapshot(self.elapsed, self.step, metrics, is_train))

    @property
    def elapsed(self):
        return time.time() - self.start

    def before_fit(self):
        self.start = time.time()

    def after_batch(self):
        for m in self.metrics:
            m.step()
        self._collect_metrics(True)
        self.step += 1

    def after_valid(self):
        self._collect_metrics(False)

# Displays training status through
# fastprogressbar and prints info
# about the status of training
class StatusCallback(Callback):
    def before_fit(self):
        header = ["epoch"]
        self.items = {
            "epoch": 5,
            "time": 11
        }

        for m in self.learner.recorder.metrics:
            if m.do_reporting:
                self.items[m] = max(len(m.name), 5)
                header.append(m.name)

        header.append("time")

        self.learner.mb.write(header, table=True)

    def after_valid(self):
        data = [f"{self.learner.epoch:<5}"]

        for m in self.learner.recorder.metrics:
            if m.do_reporting:
                val = m.value
                if not math.isfinite(val):
                    warnings.warn(f"NaN metric: {m.name}")
                    val = 0.0
                num_dig = val // 10 + 1
                max_len = self.items[m]
                data.append(f"{val:<{max_len}.{max(int(max_len - num_dig - 2), 1)}f}")

        data.append(format_time(self.learner.recorder.elapsed))

        self.learner.mb.write(data, table=True)

# Save metrics to tensorboard
class TensorboardCallback(Callback):
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.dir = log_dir

    def before_fit(self):
        self.writer = SummaryWriter(log_dir=self.dir)

    def after_fit(self):
        self.writer.close()

    def after_batch(self):
        for m in self.learner.recorder.metrics:
            if m.is_train:
                self.writer.add_scalar(m.name, m.value, self.learner.recorder.step)

        self.writer.flush()

    def after_valid(self):
        for m in self.learner.recorder.metrics:
            if not m.is_train:
                self.writer.add_scalar(m.name, m.value, self.learner.recorder.step)

        self.writer.flush()

# Save best model
class SaveBestCallback(Callback):
    def __init__(self, path: str, metric='valid loss'):
        super().__init__()
        self.path = path
        self.best = float('inf')
        self.metric = metric

    def after_valid(self):
        for m in self.learner.recorder.metrics:
            if m.name == self.metric:
                if m.value < self.best:
                    self.learner.save(self.path, silent=True)
                    self.best = m.value