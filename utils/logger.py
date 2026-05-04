"""
utils/logger.py

Logging utilities for MinecraftAI project.
Handles console logging, file logging, and TensorBoard visualization.

Author: MinecraftAI Senior Dev Team
Version: 1.0
"""

import os
from datetime import datetime

try:
    from ..config import LOG_DIR, VERBOSE
except ImportError:
    from config import LOG_DIR, VERBOSE

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# =========================
# Logger Class
# =========================
class Logger:
    """
    Logger for training/inference metrics.
    Supports:
    - Console logging
    - File logging
    - TensorBoard integration

    Additional methods:
    - log_scheduler_step(scheduler_name, value, step): Log scheduler values (e.g., LR, epsilon) for schedulers.
    """
    def __init__(self, experiment_name=None, flush_secs=30, max_queue=10):
        # Experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_name = experiment_name or f"run_{timestamp}"
        self.log_path = os.path.join(LOG_DIR, self.experiment_name)
        os.makedirs(self.log_path, exist_ok=True)

        # TensorBoard writer with performance parameters
        try:
            self.writer = SummaryWriter(self.log_path, flush_secs=flush_secs, max_queue=max_queue) if SummaryWriter else None
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard SummaryWriter: {e}")
            self.writer = None

        # File logging with persistent file handle
        self.log_file = os.path.join(self.log_path, "log.txt")
        try:
            self._log_fh = open(self.log_file, "a", buffering=1)  # line buffered
            self._log_fh.write(f"=== Logging started at {timestamp} for experiment '{self.experiment_name}' ===\n")
        except Exception as e:
            print(f"Warning: Failed to open log file {self.log_file} for writing: {e}")
            self._log_fh = None

    # =========================
    # Log scalar metrics
    # =========================
    def log_scalar(self, tag, value, step):
        """
        Log a scalar value to TensorBoard and file
        """
        # Ensure value is a float for formatting
        try:
            val_float = float(value)
        except (ValueError, TypeError):
            val_float = value  # fallback: log as is

        # TensorBoard
        if self.writer is not None:
            try:
                self.writer.add_scalar(tag, val_float, step)
            except Exception as e:
                print(f"Warning: Failed to write scalar to TensorBoard: {e}")

        # File log
        if self._log_fh is not None:
            try:
                if isinstance(val_float, float):
                    self._log_fh.write(f"[{step}] {tag}: {val_float:.4f}\n")
                else:
                    self._log_fh.write(f"[{step}] {tag}: {val_float}\n")
            except Exception as e:
                print(f"Warning: Failed to write to log file: {e}")

        # Console
        if VERBOSE:
            if isinstance(val_float, float):
                print(f"[{step}] {tag}: {val_float:.4f}")
            else:
                print(f"[{step}] {tag}: {val_float}")

    def log(self, tag, value=None, step=0):
        """
        Compatibility wrapper used by the agent and schedulers.
        """
        if value is None:
            value = tag
            tag = "message"
        self.log_scalar(str(tag), value, step)

    def info(self, message):
        self._write_text("INFO", message)

    def warning(self, message):
        self._write_text("WARNING", message)

    def error(self, message):
        self._write_text("ERROR", message)

    def log_scheduler_step(self, scheduler_name, value=None, step=0):
        """
        Log scheduler values while accepting both structured and legacy call forms.
        """
        if value is None:
            self._write_text("SCHEDULER", scheduler_name)
            return
        self.log_scalar(f"scheduler/{scheduler_name}", value, step)

    def _write_text(self, level, message):
        line = f"[{level}] {message}"
        if self._log_fh is not None:
            try:
                self._log_fh.write(line + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to log file: {e}")
        if VERBOSE:
            print(line)

    # =========================
    # Log dictionary of metrics
    # =========================
    def log_metrics(self, metrics_dict, step, substep=None):
        """
        Log multiple metrics at once
        metrics_dict: {metric_name: value}
        Supports nested dictionaries by flattening keys with dot notation.
        """
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_metrics = flatten_dict(metrics_dict)
        effective_step = step if substep is None else (step * 1_000_000) + substep
        for key, value in flat_metrics.items():
            self.log_scalar(key, value, effective_step)

    # =========================
    # Close Logger
    # =========================
    def close(self):
        """
        Close the TensorBoard writer and file handle
        """
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                print(f"Warning: Failed to close TensorBoard writer: {e}")
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception as e:
                print(f"Warning: Failed to close log file handle: {e}")

# =========================
# Example Usage (Testing)
# =========================
if __name__ == "__main__":
    logger = Logger("test_experiment")
    for step in range(5):
        logger.log_scalar("reward", 10 * step, step)
        logger.log_scalar("epsilon", 1.0 - 0.1 * step, step)
    logger.close()
    print(f"Logs saved in {logger.log_path}")
