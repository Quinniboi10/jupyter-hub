from .helper import *

# Base class for a metric
class Metric():
    is_train = True # If true, the metric will be recorded every batch, otherwise after every epoch
    do_reporting = True # If true, the metric will be printed by StatusCallback
    name = "Metric base class"

    def set_learner(self, learner):
        self.learner = learner

    # Called every batch
    def step(self):
        pass

    @property
    def value(self):
        pass


class LRMetric(Metric):
    is_train = True
    do_reporting = False
    name = "lr"

    @property
    def value(self):
        return self.learner.lr

class TrainLossMetric(Metric):
    is_train = True
    do_reporting = True
    name = "train loss"

    def __init__(self, moving_avg_div=10):
        self.div = moving_avg_div

    def set_learner(self, learner):
        self.learner = learner
        self.loss = ExpMovingAvg(learner.train_bs / self.div)

    def step(self):
        self.loss.step(self.learner.train_loss)

    @property
    def value(self):
        return self.loss.get()

class ValidLossMetric(Metric):
    is_train = False
    do_reporting = True
    name = "valid loss"

    @property
    def value(self):
        return self.learner.valid_loss