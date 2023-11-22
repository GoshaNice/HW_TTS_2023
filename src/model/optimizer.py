import torch
import numpy as np


class ScheduledOptim:
    def __init__(
        self,
        model,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0,
        n_warmup_steps=4000,
        current_step=1,
        d_model=256,
    ):
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.n_warmup_steps = n_warmup_steps
        self.current_step = current_step
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def state_dict(self):
        return self._optimizer.state_dict()

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        return lr

    def get_last_lr(self):
        return self.init_lr * self._get_lr_scale()

    def _update_learning_rate(self):
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
