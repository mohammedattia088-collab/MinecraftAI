"""
utils/scheduler.py

Scheduler utilities for MinecraftAI project.
Handles epsilon-greedy decay and learning rate scheduling.

Author: MinecraftAI Senior Dev Team
Version: 1.0
"""

import math
from typing import Optional, Any

try:
    from ..config import TRAINING_PARAMS
except ImportError:
    from config import TRAINING_PARAMS

# =========================
# Epsilon Decay Scheduler
# =========================
class EpsilonScheduler:
    """
    Linear or exponential epsilon decay scheduler for RL exploration.
    """
    def __init__(self,
                 start: float = None,
                 end: float = None,
                 decay: float = None,
                 mode: str = 'linear',
                 logger: Optional[Any] = None) -> None:
        self.start = start if start is not None else TRAINING_PARAMS.get('epsilon_start', 1.0)
        self.end = end if end is not None else TRAINING_PARAMS.get('epsilon_end', 0.1)
        self.decay = decay if decay is not None else TRAINING_PARAMS.get('epsilon_decay', 0.001)
        self.mode = mode.lower()
        self.epsilon = self.start
        self.step_count = 0
        self.logger = logger

        # Clamp values to valid ranges
        self.start = max(0.0, min(self.start, 1.0))
        self.end = max(0.0, min(self.end, self.start))
        self.epsilon = self.start

    def step(self) -> float:
        """
        Update epsilon value based on the decay schedule.
        Returns the new epsilon.
        """
        self.step_count += 1

        if self.mode == 'linear':
            self.epsilon -= self.decay
            if self.epsilon < self.end:
                self.epsilon = self.end
        elif self.mode == 'exponential':
            self.epsilon = self.end + (self.start - self.end) * math.exp(-self.decay * self.step_count)
            if self.epsilon < self.end:
                self.epsilon = self.end
            elif self.epsilon > self.start:
                self.epsilon = self.start
        else:
            raise ValueError(f"Unsupported decay mode: {self.mode}")

        if self.logger:
            self.logger.log_scheduler_step("epsilon", self.epsilon, self.step_count)

        return self.epsilon

    def get_epsilon(self, step: Optional[int] = None) -> float:
        """
        Return epsilon for a step without requiring callers to mutate the scheduler manually.
        """
        if step is not None:
            self.step_count = max(0, int(step))
            if self.mode == 'linear':
                self.epsilon = max(self.end, self.start - self.decay * self.step_count)
            elif self.mode == 'exponential':
                self.epsilon = self.end + (self.start - self.end) * math.exp(-self.decay * self.step_count)
                self.epsilon = max(self.end, min(self.start, self.epsilon))
            else:
                raise ValueError(f"Unsupported decay mode: {self.mode}")
        return self.epsilon

    def state_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "decay": self.decay,
            "mode": self.mode,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.start = float(state.get("start", self.start))
        self.end = float(state.get("end", self.end))
        self.decay = float(state.get("decay", self.decay))
        self.mode = str(state.get("mode", self.mode)).lower()
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.step_count = int(state.get("step_count", self.step_count))

    def reset(self) -> None:
        """
        Reset epsilon to the starting value.
        """
        self.epsilon = self.start
        self.step_count = 0
        if self.logger:
            self.logger.log_scheduler_step("epsilon", self.epsilon, self.step_count)

# =========================
# Learning Rate Scheduler
# =========================
class LRScheduler:
    """
    Learning rate scheduler for PyTorch optimizer.
    Supports step decay and exponential decay.
    """
    def __init__(self,
                 optimizer: Any,
                 initial_lr: Optional[float] = None,
                 init_lr: Optional[float] = None,
                 mode: str = 'step',
                 step_size: int = 1000,
                 gamma: float = 0.99,
                 logger: Optional[Any] = None) -> None:
        self.optimizer = optimizer
        if initial_lr is None:
            initial_lr = init_lr
        if initial_lr is None:
            raise ValueError("initial_lr or init_lr is required")
        self.initial_lr = max(1e-10, initial_lr)  # minimum lr to avoid zero or negative
        self.mode = mode.lower()
        self.step_size = step_size
        self.gamma = gamma
        self.step_count = 0
        self.logger = logger

        # Initialize optimizer learning rates to initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

    def step(self) -> None:
        """
        Update the learning rate according to the schedule.
        """
        self.step_count += 1
        if self.mode == 'step':
            if self.step_count % self.step_size == 0:
                for param_group in self.optimizer.param_groups:
                    new_lr = param_group['lr'] * self.gamma
                    # Clamp to a minimum learning rate
                    param_group['lr'] = max(new_lr, 1e-10)
        elif self.mode == 'exponential':
            lr = self.initial_lr * (self.gamma ** self.step_count)
            lr = max(lr, 1e-10)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            raise ValueError(f"Unsupported LR mode: {self.mode}")

        if self.logger:
            current_lr = self.get_lr()
            self.logger.log_scheduler_step("learning_rate", current_lr, self.step_count)

    def get_lr(self) -> float:
        """
        Return current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']

    def get_last_lr(self) -> list[float]:
        return [self.get_lr()]

    def state_dict(self) -> dict:
        return {
            "initial_lr": self.initial_lr,
            "mode": self.mode,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "step_count": self.step_count,
            "current_lr": self.get_lr(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.initial_lr = float(state.get("initial_lr", self.initial_lr))
        self.mode = str(state.get("mode", self.mode)).lower()
        self.step_size = int(state.get("step_size", self.step_size))
        self.gamma = float(state.get("gamma", self.gamma))
        self.step_count = int(state.get("step_count", self.step_count))
        current_lr = float(state.get("current_lr", self.get_lr()))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(current_lr, 1e-10)

    def reset(self) -> None:
        """
        Reset learning rate to initial value.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        self.step_count = 0
        if self.logger:
            self.logger.log_scheduler_step("learning_rate", self.initial_lr, self.step_count)

# =========================
# Example Usage (Testing)
# =========================
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SchedulerTest")

    class DemoOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 0.01}]

    print("Testing EpsilonScheduler with Logger:")
    eps_sched = EpsilonScheduler(start=1.0, end=0.1, decay=0.1, mode='linear', logger=logger)
    for i in range(12):
        eps = eps_sched.step()
        print(f"Step {i}: epsilon={eps:.3f}")

    print("\nTesting LRScheduler with Logger:")
    demo_optimizer = DemoOptimizer()
    lr_sched = LRScheduler(demo_optimizer, initial_lr=0.01, mode='step', step_size=3, gamma=0.5, logger=logger)
    for i in range(10):
        lr_sched.step()
        current_lr = lr_sched.get_lr()
        print(f"Step {i}: lr={current_lr:.5f}")
