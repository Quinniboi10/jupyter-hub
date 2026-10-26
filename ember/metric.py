import torch

from .helper import *

from enum import Enum, auto

class State(Enum):
    TRAIN = auto()
    VALID = auto()

# Base class for a metric
class Metric():
    state = State.TRAIN # State of the learner in which to run step()
    do_reporting = True # If True, the metric will be printed by StatusCallback
    name = "Metric base class"

    def __eq__(self, other) -> bool:
        assert isinstance(other, Metric)
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def set_learner(self, learner):
        self.learner = learner

    # Called every batch
    def step(self):
        pass

    @property
    def value(self):
        raise NotImplementedError()

# Base class for metrics that implement some underlying exponential moving avgerage
class ExpAvgMetric(Metric):
    name = "Average base class"

    def __init__(self, moving_avg_div=10):
        self.div = moving_avg_div

    def set_learner(self, learner):
        self.learner = learner
        self.avg = ExpMovingAvg(learner.train_bs / self.div)

    def step(self):
        raise NotImplementedError()

    @property
    def value(self):
        return self.avg.get()


class LRMetric(Metric):
    state = State.TRAIN
    do_reporting = False
    name = "lr"

    @property
    def value(self):
        return self.learner.lr

class TrainLossMetric(ExpAvgMetric):
    state = State.TRAIN
    do_reporting = True
    name = "train loss"

    def step(self):
        self.avg.step(self.learner.train_loss)

class ValidLossMetric(Metric):
    state = State.VALID
    do_reporting = True
    name = "valid loss"

    @property
    def value(self):
        return self.learner.valid_loss

class OutputVarianceMetric(ExpAvgMetric):
    state = State.TRAIN
    do_reporting = True
    name = "pred variance"

    def step(self):
        self.avg.step(float(torch.var(self.learner.preds, dim=1).mean().detach()))