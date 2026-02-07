import torch

from .callback import *
from .helper import *
from .metric import *

from fastprogress.fastprogress import master_bar, progress_bar
import warnings

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Learner():
    def __init__(self, model, optimizer, train_dls, valid_dls, loss_fn, metrics=None, compile_model=True, clip_grads=None):
        self.model = model.to(device)
        if compile_model:
            self.model = torch.compile(self.model)
        self.optim = optimizer(self.model.parameters())

        self.dl = train_dls
        self.valid_dl = valid_dls

        self.loss_fn = loss_fn
        self.clip_grads = clip_grads

        self.train_bs = len(train_dls)
        self.valid_bs = len(valid_dls)

        self.n_epochs = None

        self.schedule = None

        self.cbs = []

        self.loss_fn.learner = self
        self.device = device

        # For complex loss functions or
        # other things that need to use
        # the input or output
        self.x = None
        self.y = None

        ### Things for callbacks to use
        # Updated every batch
        self.lr = None
        self.train_loss = None

        # Updated every epoch
        self.epoch = 0
        self.valid_loss = None

        # Add callbacks
        if metrics is None:
            metrics = [LRMetric(), TrainLossMetric(), ValidLossMetric()]

        self.add_cb(RecorderCallback(metrics))
        self.add_cb(TrainEvalCallback())
        self.add_cb(StatusCallback())

    def __call__(self, *args):
        return self.model(*args)

    def to(self, device):
        self.device = device
        self.model.to(device)
        if self.x is not None:
            self.x.to(device)
        if self.y is not None:
            self.y.to(device)

    def _run_epoch(self, curr_epoch):
        for batch, (self.x, self.y) in enumerate(progress_bar(self.dl, parent=self.mb)):
            lr = self.schedule.get_last_lr()
            assert len(lr) == 1, "Schedule LR must have a length of one!"
            self.lr = lr[0]

            self._run_cbs("before_batch")

            self.x = self.x.to(self.device)
            self.y = self.y.to(self.device)

            # Forward
            with torch.autocast(device_type=self.device):
                z = self.model(self.x)
                loss = self.loss_fn(z, self.y)

            self.train_loss = loss.item()

            # Backward
            self.optim.zero_grad()
            loss.backward()
            if self.clip_grads is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads)
            self.optim.step()
            self.schedule.step()

            self._run_cbs("after_batch")

    def _run_valid(self):
        total = 0
        with torch.no_grad(), torch.autocast(device_type=self.device):
            for self.x, self.y in progress_bar(self.valid_dl, parent=self.mb):
                self.x = self.x.to(self.device)
                self.y = self.y.to(self.device)

                # Forward
                z = self.model(self.x)
                total += self.loss_fn(z, self.y).item()
        self.valid_loss = total / self.valid_bs

    def fit_one_cycle(self, num_epochs, max_lr=1e-3, div=25, final_div=1e5) -> None:
        if max_lr >= 1e-2:
            warnings.warn("fit_one_cycle has high max_lr which could lead to gradient explosions! If loss becomes NaN, try again with a lower max LR.")
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, steps_per_epoch=len(self.dl), epochs=num_epochs,
            max_lr=max_lr, div_factor=div, final_div_factor=final_div
        )
        self.fit(num_epochs, self_schedule=True)

    def fit(self, num_epochs, self_schedule=False):
        self.mb = master_bar(range(num_epochs))
        if self_schedule == False:
            self.schedule = torch.optim.lr_scheduler.ConstantLR(
                self.optim,
                factor=1.0
            )

        self._run_cbs("before_fit")

        try:
            for _ in self.mb:
                self._run_cbs("before_train")
                self._run_epoch(self.epoch)
                self._run_cbs("after_train")

                self._run_cbs("before_valid")
                self._run_valid()
                self._run_cbs("after_valid")

                self.epoch += 1

        except KeyboardInterrupt:
            print("Training cancelled. Exiting...")

        self._run_cbs("after_fit")

    def save(self, path: str):
        if path[-4:] != ".pth":
            path += ".pth"
        torch.save(self.model, path)

    def add_cb(self, cb):
        cb.set_learner(self)
        for existing in self.cbs:
            if existing.name == cb.name:
                raise RuntimeError("A callback with the same name already exists!")
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def _run_cbs(self, state: str):
        for cb in self.cbs:
            if hasattr(cb, state):
                getattr(cb, state)()