import torch

from .helper import *

# Base class for a metric
class Metric():
    is_train = False    # If True, the metric will be recorded every batch, otherwise after every epoch
    do_reporting = True # If True, the metric will be printed by StatusCallback
    name = "Metric base class"

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
    is_train = True
    do_reporting = False
    name = "lr"

    @property
    def value(self):
        return self.learner.lr

class TrainLossMetric(ExpAvgMetric):
    is_train = True
    do_reporting = True
    name = "train loss"

    def step(self):
        self.avg.step(self.learner.train_loss)

class ValidLossMetric(Metric):
    is_train = False
    do_reporting = True
    name = "valid loss"

    @property
    def value(self):
        return self.learner.valid_loss

class OutputVarianceMetric(ExpAvgMetric):
    is_train = True
    do_reporting = True
    name = "pred variance"

    def step(self):
        self.avg.step(float(torch.var(self.learner.preds, dim=1).mean().detach()))