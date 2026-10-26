from torch.utils.data import DataLoader
from torch import nn
import torch

from .callback import *
from .helper import *
from .metric import *

from fastprogress.fastprogress import master_bar, progress_bar
import warnings
import math
import pathlib
from typing import cast, Optional

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # pyright: ignore[reportOptionalMemberAccess]

class Learner():
    # Type hints on required CBs
    recorder: RecorderCallback

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_dl: DataLoader, valid_dl: DataLoader, loss_fn: nn.Module, metrics: Optional[list[Metric]]=None, cbs: Optional[list[Callback]]=None, clip_grads: Optional[float]=0, accum: int=1):
        """
        Args:
            model: the PyTorch model to train
            optimizer: the PyTorch optimizer to run
            train_dl: the PyTorch train dataloader
            valid_dl: the PyTorch valid dataloader
            loss_fn: An extension of nn.Module that implements forward(preds, targets)
            metrics: Which metrics to use apart from LR, train loss, valid loss
            cbs: Which callbacks to attach after Recorder, TrainEval, State; defaults to Status (None)
            clip_grads: Clip gradients before optimizer.step(); defaults to off (None)
            accum: How many batches to average loss across before running optimizer.step(). Simulates true_bs = train_bs * accum
        """
        self.model = model
        self.optim = optimizer

        self.dl = train_dl
        self.valid_dl = valid_dl

        self.loss_fn = loss_fn
        self.clip_grads = clip_grads
        self.accum = accum

        self.num_train_batches = len(train_dl)
        self.num_valid_batches = len(valid_dl)

        assert isinstance(train_dl.batch_size, int)
        assert isinstance(valid_dl.batch_size, int)

        self.train_bs = train_dl.batch_size
        self.valid_bs = valid_dl.batch_size


        self.n_epochs = cast(int, None)

        self.schedule = cast(torch.optim.lr_scheduler.LRScheduler, None)

        self.cbs: list[Callback] = []

        self.loss_fn.learner = self
        self.device = device

        assert train_dl.batch_size

        if accum > 1:
            print("Simulating batch size of", train_dl.batch_size * accum)

        # Shows the current state of the learner as it trains
        self.state = cast(State, None)

        # For complex loss functions or
        # other things that need to use
        # the input or output
        self.x = cast(torch.Tensor, None)
        self.y = cast(torch.Tensor, None)
        self.preds = cast(torch.Tensor, None)

        ### Things for callbacks to use
        # Updated every batch
        self.lr = cast(float, None)
        self.train_loss = cast(float, None)

        # Updated every epoch
        self.epoch = 0
        self.valid_loss = cast(float, None)

        # Add callbacks and metrics
        if metrics is None:
            metrics = []

        # Always-on metrics
        for m in reversed([LRMetric(), TrainLossMetric(), ValidLossMetric()]):
            if m not in metrics:
                metrics.insert(0, m)

        if cbs is None:
            cbs = [StatusCallback()]

        for cb in cbs:
            self.add_cb(cb)

        # Try to add required callbacks and if they already exist then just keep moving
        for cb in [RecorderCallback(metrics), TrainEvalCallback(), StateCallback()]:
            try:
                self.add_cb(cb)
            except RuntimeError:
                pass

        # Move to device
        self.to(device)

    def __call__(self, *args):
        return self.model(*args)

    def to(self, device):
        self.device = device
        self.model.to(device)
        if self.x is not None:
            self.x = self.x.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.preds is not None:
            self.preds = self.preds.to(device)

    def _run_epoch(self, curr_epoch):
        self.optim.zero_grad()
        for batch, (self.x, self.y) in enumerate(progress_bar(self.dl, parent=self.mb)):
            lr = self.schedule.get_last_lr()
            assert len(lr) == 1, "Schedule LR must have a length of one!"
            self.lr = lr[0]
            assert isinstance(self.lr, float), f"LR Scheduler get_last_lr() expected return type list[float] with length 1, got {type(self.lr)}"

            self._run_cbs("before_batch")

            self.x = self.x.to(self.device, non_blocking=True)
            self.y = self.y.to(self.device, non_blocking=True)

            # Forward
            with torch.autocast(device_type=self.device):
                self.preds = self.model(self.x)
                raw_loss = self.loss_fn(self.preds, self.y)
                loss = raw_loss / self.accum

            # Report the unscaled loss so train/valid are comparable
            self.train_loss = raw_loss.item()

            # Backward
            loss.backward()
            if (batch + 1) % self.accum == 0:
                if self.clip_grads is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                self.optim.step()
                self.schedule.step()
                self.optim.zero_grad()

            self._run_cbs("after_batch")

        remainder = (batch + 1) % self.accum
        if remainder != 0:
            if self.clip_grads is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads)
            self.optim.step()
            self.schedule.step()
            self.optim.zero_grad()

    def _run_valid(self):
        total = 0
        with torch.no_grad(), torch.autocast(device_type=self.device):
            for self.x, self.y in progress_bar(self.valid_dl, parent=self.mb):
                self._run_cbs("before_batch")

                self.x = self.x.to(self.device, non_blocking=True)
                self.y = self.y.to(self.device, non_blocking=True)

                # Forward
                self.preds = self.model(self.x)
                total += self.loss_fn(self.preds, self.y).item()
                self._run_cbs("after_batch")
        self.valid_loss = total / self.num_valid_batches

    def fit_one_cycle(self, num_epochs, max_lr=1e-3, div=25, final_div=1e5) -> None:
        if max_lr >= 1e-2:
            warnings.warn("fit_one_cycle has high max_lr which could lead to gradient explosions! If loss becomes NaN, try again with a lower max LR.")
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, steps_per_epoch=math.ceil(len(self.dl) / self.accum), epochs=num_epochs,
            max_lr=max_lr, div_factor=div, final_div_factor=final_div
        )
        self.fit(num_epochs, self_schedule=True)

    def fit(self, num_epochs, self_schedule=False):
        assert num_epochs > 0, "Cannot train for less than 1 epoch"
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

    def save(self, path: str, silent=False):
        p = pathlib.Path(path)
        if p.suffix != ".pt" and p.suffix != ".pth":
            p = p.with_suffix(".pt")
        p.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), str(p))

        if not silent:
            print(f"Model saved to {p}")

    def export(self, path: str, *args):
        p = pathlib.Path(path)
        if p.suffix != ".pt2":
            p = p.with_suffix(".pt2")
        p.parent.mkdir(parents=True, exist_ok=True)

        if not args:
            x, _ = next(iter(self.valid_dl))
            args = (x,)
        exported = torch.export.export(self.model, args)
        torch.export.save(exported, str(p))

        print(f"Model exported to {p}")

    def add_cb(self, cb):
        cb.set_learner(self)
        for existing in self.cbs:
            if existing.name == cb.name:
                raise RuntimeError("A callback with the same name already exists! It's possible it was added previously, or by default.")
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def _run_cbs(self, state: str):
        for cb in self.cbs:
            if hasattr(cb, state):
                getattr(cb, state)()

    def param_info(self, sort_by: Literal["order", "count"]="count") -> None:
        """
        Prints information about the model's parameters
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total:     {total:,}")
        print(f"Trainable: {trainable:,}")

        rows: list[list[str|int]] = []
        for name, p in self.model.named_parameters():
            rows.append([name, p.numel()])

        if sort_by == "count":
            rows.sort(key=lambda a: a[1], reverse=True)

        for i in range(len(rows)):
            rows[i][1] = f"{rows[i][1]:,}"

        rows.insert(0, ["Name", "Parameters"])

        tableify(rows)