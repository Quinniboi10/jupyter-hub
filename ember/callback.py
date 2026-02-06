import time
from enum import Enum
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .helper import *

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

class Metrics(Enum):
    lr = "lr"
    train_loss = "train loss"
    valid_loss = "valid loss"

# Metrics updated during training (every batch)
train_metrics = [Metrics.lr, Metrics.train_loss]

# Overarching class from which all other
# callbacks must inheret
class Callback():
    def __init__(self):
        self.learner = None

    def __repr__(self):
        return type(self).__name__

    @property
    def name(self):
        return camel_to_snake(self.__repr__())

    def _before_fit(self):
        pass
    def _before_train(self):
        pass
    def _before_valid(self):
        pass
    def _before_batch(self):
        pass
    def _after_fit(self):
        pass
    def _after_train(self):
        pass
    def _after_valid(self):
        pass
    def _after_batch(self):
        pass

# Toggles a model between train
# and eval mode. This callback was
# more of a proof-of-concept, not
# a neccesity
class TrainEvalCallback(Callback):
    def _before_train(self):
        self.learner.model.train()
    def _before_valid(self):
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
    class Snapshot():
        def __init__(self, elapsed: float, step: int, metrics: dict, is_train: bool):
            self.elapsed = elapsed
            self.step = step
            self.metrics = metrics
            self.is_train = is_train

        def __repr__(self):
            return f"time-{format_time(self.elapsed)}  step-{self.step}  metrics-{self.metrics}  training-{self.is_train}"

    def __init__(self, metrics):
        super().__init__()
        self.metrics = List(metrics)
        self.metrics.ensure(Metrics.lr, Metrics.train_loss, Metrics.valid_loss)

        self.step = 0
        self.total_loss = 0
        self.smooth_loss = ExpMovingAvg(30)

        self.data = []

        self.start = None

    @property
    def name(self):
        return "recorder"

    def plot_loss(self):
        x = range(0, self.step)
        y = []
        valid_x = []
        valid_y = []

        for i in range(0, len(self.data)):
            s = self.data[i]
            if s.is_train:
                y.append(s.metrics[Metrics.train_loss])
            elif Metrics.valid_loss in self.metrics:
                valid_x.append(i)
                valid_y.append(s.metrics[Metrics.valid_loss])

        print(f"Found {len(x)} train and {len(valid_x)} valid snapshots")

        plt.plot(x, y)
        plt.plot(valid_x, valid_y)

        plt.xlabel("Batch")
        plt.ylabel("Loss (smoothed)")
        plt.title("Training loss")
        plt.grid(True)

        plt.show()

    def get_metric(self, m: Metrics) -> float|None:
        if m == Metrics.lr:
            return self.learner.lr
        if m == Metrics.train_loss:
            return self.avg_loss
        if m == Metrics.valid_loss:
            return self.learner.valid_loss
        return None

    @property
    def avg_loss(self):
        return self.total_loss / max(1, self.step)

    @property
    def elapsed(self):
        return time.time() - self.start

    def _before_fit(self):
        self.start = time.time()

    def _after_batch(self):
        metrics = {}

        loss = self.learner.train_loss
        self.total_loss += loss
        self.smooth_loss.step(loss)

        for m in self.metrics:
            if m in train_metrics:
                value = self.get_metric(m)
                if value is not None:
                    metrics[m] = value

        self.data.append(self.Snapshot(self.elapsed, self.step, metrics, True))
        self.step += 1

    def _after_valid(self):
        metrics = {}

        for m in self.metrics:
            if m not in train_metrics:
                value = self.get_metric(m)
                if value is not None:
                    metrics[m] = value

        self.data.append(self.Snapshot(self.elapsed, self.step, metrics, False))

# Displays training status through
# fastprogressbar and prints info
# about the status of training
printable_metrics = [Metrics.train_loss, Metrics.valid_loss] # Metrics to allow to be printed
class StatusCallback(Callback):
    def _before_fit(self):
        header = ["epoch"]
        self.items = {
            "epoch": 5,
            "time": 11
        }

        for m in self.learner.recorder.metrics:
            if m in printable_metrics:
                self.items[m] = max(len(m.value), 5)
                header.append(m.value)

        header.append("time")

        self.learner.mb.write(header, table=True)

    def _after_valid(self):
        data = [f"{self.learner.epoch:<5}"]

        for m in self.learner.recorder.metrics:
            if m in printable_metrics:
                val = self.learner.recorder.get_metric(m)
                num_dig = val // 10 + 1
                max_len = self.items[m]
                data.append(f"{val:<{max_len}.{int(max_len - num_dig - 2)}f}")

        data.append(format_time(self.learner.recorder.elapsed))

        self.learner.mb.write(data, table=True)

# Save metrics to tensorboard
class TensorboardCallback(Callback):
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.dir = log_dir

    def _before_fit(self):
        self.writer = SummaryWriter(log_dir=self.dir)

    def _after_fit(self):
        self.writer.close()

    def _after_batch(self):
        for m in self.learner.recorder.metrics:
            if m in train_metrics:
                self.writer.add_scalar(m.value, self.learner.recorder.get_metric(m), self.learner.recorder.step)

        self.writer.flush()

    def _after_valid(self):
        for m in self.learner.recorder.metrics:
            if m not in train_metrics:
                self.writer.add_scalar(m.value, self.learner.recorder.get_metric(m), self.learner.recorder.step)

        self.writer.flush()