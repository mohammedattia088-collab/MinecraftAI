"""
perception/visual.py

Perception module for MinecraftAI project.
Handles all visual input processing, CNN feature extraction, and frame stacking.
Designed for compatibility with M1 Mac (MPS) and dynamic resolution handling.

Author: MinecraftAI Senior Dev Team
Version: 1.0
"""

from __future__ import annotations

import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional in lightweight bridge runs
    torch = None

    class _MissingTorchModule:
        class Module:
            """Stand-in base class used when torch is not installed."""

        def __getattr__(self, name: str) -> Any:
            raise ImportError("torch is required for CNNFeatureExtractor")

    nn = _MissingTorchModule()
    F = _MissingTorchModule()

try:
    import cv2
except Exception:  # pragma: no cover - optional in lightweight bridge runs
    cv2 = None

try:
    from ..config import SCREEN_WIDTH, SCREEN_HEIGHT, NUM_CHANNELS, FRAME_STACK, MPS_DEVICE
except ImportError:
    from config import SCREEN_WIDTH, SCREEN_HEIGHT, NUM_CHANNELS, FRAME_STACK, MPS_DEVICE

# =========================
# Device configuration
# =========================
class _Device:
    def __init__(self, device_type: str) -> None:
        self.type = device_type

    def __str__(self) -> str:
        return self.type


DEVICE = (
    torch.device("mps" if MPS_DEVICE and torch.backends.mps.is_available() else "cpu")
    if torch is not None
    else _Device("cpu")
)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int32)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int32)
    return frame[y_idx][:, x_idx]


@dataclass
class VisualObservation:
    brightness: float
    contrast: float
    saturation: float
    edge_density: float
    lava_ratio: float
    foliage_ratio: float
    sky_water_ratio: float
    motion_score: float
    novelty_score: float
    dominant_rgb: tuple[int, int, int]
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "edge_density": self.edge_density,
            "lava_ratio": self.lava_ratio,
            "foliage_ratio": self.foliage_ratio,
            "sky_water_ratio": self.sky_water_ratio,
            "motion_score": self.motion_score,
            "novelty_score": self.novelty_score,
            "dominant_rgb": list(self.dominant_rgb),
            "timestamp": self.timestamp,
        }


def analyze_frame(frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> VisualObservation:
    """
    Cheap visual analysis for action selection and reward shaping.

    It deliberately uses a downsampled frame so the learner can keep this in the
    hot path without turning every step into a computer vision batch job.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty numpy array")

    if frame.ndim == 2:
        rgb = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.shape[2] == 4:
        rgb = frame[:, :, :3]
    else:
        rgb = frame[:, :, :3]

    sample = resize_frame(rgb, 160, 90)
    if cv2 is not None:
        hsv = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    else:
        hsv = None
        gray = np.mean(sample, axis=2).astype(np.uint8)

    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)
    if cv2 is not None and hsv is not None:
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        edges = cv2.Canny(gray, 60, 140)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        lava_mask = cv2.inRange(hsv, np.array([4, 120, 120]), np.array([25, 255, 255]))
        foliage_mask = cv2.inRange(hsv, np.array([35, 45, 35]), np.array([95, 255, 210]))
        sky_water_mask = cv2.inRange(hsv, np.array([90, 35, 60]), np.array([130, 255, 255]))
        lava_ratio = float(np.count_nonzero(lava_mask) / lava_mask.size)
        foliage_ratio = float(np.count_nonzero(foliage_mask) / foliage_mask.size)
        sky_water_ratio = float(np.count_nonzero(sky_water_mask) / sky_water_mask.size)
    else:
        channel_max = np.max(sample, axis=2).astype(np.float32)
        channel_min = np.min(sample, axis=2).astype(np.float32)
        saturation = float(np.mean((channel_max - channel_min) / np.maximum(1.0, channel_max)))
        grad_y, grad_x = np.gradient(gray.astype(np.float32))
        edge_density = float(np.mean(np.sqrt(grad_x * grad_x + grad_y * grad_y) > 28.0))
        red = sample[:, :, 0].astype(np.int16)
        green = sample[:, :, 1].astype(np.int16)
        blue = sample[:, :, 2].astype(np.int16)
        lava_ratio = float(np.mean((red > 160) & (green > 55) & (green < 170) & (blue < 90)))
        foliage_ratio = float(np.mean((green > red + 20) & (green > blue + 10) & (green > 55)))
        sky_water_ratio = float(np.mean((blue > red + 20) & (blue > green - 10) & (blue > 80)))

    motion_score = 0.0
    novelty_score = edge_density + contrast
    if previous_frame is not None and isinstance(previous_frame, np.ndarray) and previous_frame.size:
        prev = previous_frame
        if prev.ndim == 2:
            prev = np.repeat(prev[:, :, None], 3, axis=2)
        elif prev.shape[2] == 4:
            prev = prev[:, :, :3]
        prev_sample = resize_frame(prev[:, :, :3], 160, 90)
        diff = np.abs(sample.astype(np.int16) - prev_sample.astype(np.int16)).astype(np.uint8)
        motion_score = float(np.mean(diff) / 255.0)
        novelty_score += motion_score * 2.0

    pixels = sample.reshape(-1, 3)
    quantized = (pixels // 32).astype(np.uint8)
    bins, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = tuple(int(channel * 32 + 16) for channel in bins[int(np.argmax(counts))])

    return VisualObservation(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        edge_density=edge_density,
        lava_ratio=lava_ratio,
        foliage_ratio=foliage_ratio,
        sky_water_ratio=sky_water_ratio,
        motion_score=motion_score,
        novelty_score=float(novelty_score),
        dominant_rgb=dominant,
    )


class VisionMemory:
    """Rolling visual analytics used by the environment and autonomous trainer."""

    def __init__(self, maxlen: int = 120) -> None:
        self.maxlen = maxlen
        self.frames = deque(maxlen=2)
        self.observations: deque[VisualObservation] = deque(maxlen=maxlen)

    def add_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        previous = self.frames[-1] if self.frames else None
        observation = analyze_frame(frame, previous)
        self.frames.append(frame.copy())
        self.observations.append(observation)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        if not self.observations:
            return {
                "available": False,
                "frame_count": 0,
                "needs_attention": False,
                "rolling": {
                    "brightness": 0.0,
                    "novelty": 0.0,
                    "motion": 0.0,
                    "lava": 0.0,
                },
            }
        latest = self.observations[-1].as_dict()
        window = list(self.observations)[-min(20, len(self.observations)):]
        latest["rolling"] = {
            "brightness": float(np.mean([obs.brightness for obs in window])),
            "novelty": float(np.mean([obs.novelty_score for obs in window])),
            "motion": float(np.mean([obs.motion_score for obs in window])),
            "lava": float(np.max([obs.lava_ratio for obs in window])),
        }
        latest["needs_attention"] = (
            latest["lava_ratio"] > 0.04
            or latest["brightness"] < 0.12
            or latest["motion_score"] < 0.004
        )
        return latest


# =========================
# Preprocessing Functions
# =========================
def preprocess_frame(
    frame: np.ndarray,
    width: int = SCREEN_WIDTH,
    height: int = SCREEN_HEIGHT,
    grayscale: bool = False,
    logger: Optional[object] = None,
    log_scheduler_step: Optional[int] = None,
) -> np.ndarray:
    """
    Preprocess a raw Minecraft frame.
    - Resize to fixed width/height
    - Convert to grayscale (optional)
    - Normalize pixel values (0-1)
    - Convert to CHW format for PyTorch
    - Optionally performs normalization and transpose on GPU if available

    Parameters:
        frame: np.ndarray
            Raw input frame.
        width: int
            Target width.
        height: int
            Target height.
        grayscale: bool
            Whether to convert frame to grayscale.
        logger: Optional[object]
            If provided, logs preprocessing metrics such as frame shape and preprocessing time using log_scheduler_step.
        log_scheduler_step: Optional[int]
            Step number to log with logger, if applicable.

    Returns:
        np.ndarray: Preprocessed frame in CHW format.

    Raises:
        ValueError: If frame is None or has invalid dimensions.
    """
    if frame is None:
        raise ValueError("Input frame is None.")
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Input frame must be a numpy array, got {type(frame)}.")
    if frame.ndim not in [2, 3]:
        raise ValueError(f"Input frame must have 2 or 3 dimensions, got {frame.ndim}.")

    start_time = time.time()
    # Resize frame
    frame_resized = resize_frame(frame, width, height)

    # Grayscale conversion if needed
    if grayscale:
        if frame_resized.ndim == 3 and frame_resized.shape[2] == 3:
            if cv2 is not None:
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
            else:
                frame_resized = np.mean(frame_resized, axis=2).astype(frame_resized.dtype)
        # Add channel dimension if missing
        if frame_resized.ndim == 2:
            frame_resized = np.expand_dims(frame_resized, axis=2)

    if torch is not None and DEVICE.type in ["cuda", "mps"]:
        # Use torch for normalization and transpose on GPU
        frame_tensor = torch.from_numpy(frame_resized).to(DEVICE, non_blocking=True).float()
        frame_tensor = frame_tensor / 255.0  # Normalize
        # HWC to CHW
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        elif frame_tensor.ndim == 2:
            frame_tensor = frame_tensor.unsqueeze(0)
        frame_chw = frame_tensor.cpu().numpy()
    else:
        # Normalize
        frame_resized = frame_resized.astype(np.float32) / 255.0
        # HWC to CHW
        if frame_resized.ndim == 3:
            frame_chw = np.transpose(frame_resized, (2, 0, 1))
        elif frame_resized.ndim == 2:
            frame_chw = np.expand_dims(frame_resized, axis=0)
        else:
            raise ValueError(f"Unexpected frame_resized ndim {frame_resized.ndim}")

    elapsed = time.time() - start_time
    if logger is not None and log_scheduler_step is not None:
        logger.log_scheduler_step(
            f"Preprocessed frame shape: {frame_chw.shape}, time taken: {elapsed:.4f}s",
            step=log_scheduler_step,
        )
    elif logger is not None and hasattr(logger, "info"):
        logger.info(f"Preprocessed frame shape: {frame_chw.shape}, time taken: {elapsed:.4f}s")

    return frame_chw


# =========================
# Frame Stack Class
# =========================
class FrameStack:
    """
    Maintains a stack of the last N frames for temporal context.

    Parameters:
        stack_size: int
            Number of frames to stack.
        channels: int
            Number of channels per frame.
        width: int
            Frame width.
        height: int
            Frame height.
        grayscale: bool
            Whether frames are grayscale (affects channel count).
        logger: Optional[object]
            If provided, logs errors during frame addition.
        frame_logger: Optional[object]
            If provided, logs events about frame stacking separate from the global logger.
    """

    def __init__(
        self,
        stack_size: int = FRAME_STACK,
        channels: int = NUM_CHANNELS,
        width: int = SCREEN_WIDTH,
        height: int = SCREEN_HEIGHT,
        grayscale: bool = False,
        logger: Optional[object] = None,
        frame_logger: Optional[object] = None,
    ) -> None:
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.channels = 1 if grayscale else channels
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.logger = logger
        self.frame_logger = frame_logger
        # Initialize with zeros
        for _ in range(stack_size):
            self.frames.append(np.zeros((self.channels, height, width), dtype=np.float32))

    def add_frame(self, frame: np.ndarray, from_neoforge: bool = False, log_scheduler_step: Optional[int] = None) -> None:
        """
        Add a new frame to the stack after preprocessing.
        Handles errors gracefully to avoid crashes.

        Parameters:
            frame: np.ndarray
                Raw input frame.
            from_neoforge: bool
                Indicates if the frame is from NeoForge environment or fallback.
            log_scheduler_step: Optional[int]
                Step number to log preprocessing metrics if logger supports it.
        """
        source = "NeoForge" if from_neoforge else "Fallback"
        if self.frame_logger is not None:
            self.frame_logger.info(f"Adding new frame from {source}. Current stack size: {len(self.frames)}")
        try:
            if self._is_preprocessed_frame(frame):
                preprocessed = frame.astype(np.float32, copy=False)
            else:
                preprocessed = preprocess_frame(
                    frame,
                    self.width,
                    self.height,
                    self.grayscale,
                    logger=self.logger,
                    log_scheduler_step=log_scheduler_step,
                )
            self.frames.append(preprocessed)
            if self.frame_logger is not None:
                self.frame_logger.info(f"Frame from {source} added successfully. Stack size is now {len(self.frames)}")
        except Exception as e:
            if self.frame_logger is not None:
                self.frame_logger.error(f"Failed to add frame from {source} to stack: {e}")
            if self.logger is not None:
                self.logger.error(f"Failed to add frame from {source} to stack: {e}")
            # Append zero frame to maintain stack size and avoid errors downstream
            self.frames.append(np.zeros((self.channels, self.height, self.width), dtype=np.float32))

    def _is_preprocessed_frame(self, frame: np.ndarray) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape == (self.channels, self.height, self.width)
        )

    def get_stack(self) -> np.ndarray:
        """
        Return stacked frames in CHW format: (channels * stack_size, height, width).

        Frames are stored internally in CHW format (C, H, W). This method concatenates
        along the channel axis and returns a NumPy array suitable for direct conversion
        to a PyTorch tensor with shape (C_total, H, W).
        """
        # Ensure we have frames available
        if len(self.frames) == 0:
            raise ValueError("FrameStack contains no frames. Add at least one frame before calling get_stack().")

        # Frames are already CHW (C, H, W). Concatenate along channel axis (axis=0)
        stacked = np.concatenate(list(self.frames), axis=0)

        # Validate resulting shape
        if stacked.ndim != 3:
            raise RuntimeError(f"Unexpected stacked frames ndim: {stacked.ndim}. Expected 3 (C,H,W). Got shape {stacked.shape}.")

        return stacked


# =========================
# CNN Feature Extractor
# =========================
class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network to extract visual features from stacked Minecraft frames.
    Dynamically computes flattened feature size to avoid hardcoded values.
    Supports dynamic input channels (grayscale or RGB).

    Parameters:
        input_shape: tuple[int, int, int]
            Input tensor shape (channels, height, width), where channels can be 1 or 3 or multiples due to stacking.
    """

    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        if torch is None:
            raise ImportError("torch is required to create CNNFeatureExtractor")
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Compute dynamic feature size with a zero-valued shape probe.
        with torch.no_grad():
            probe_input = torch.zeros(1, *input_shape)
            self.feature_size = self.conv(probe_input).view(1, -1).size(1)
        self.fc = nn.Linear(self.feature_size, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN and FC layer
        Input: x (batch, channels, height, width)
        Output: feature vector (batch, 512)
        """
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x


# =========================
# Convenience Function
# =========================
def get_cnn_input_shape(
    frame_stack: "FrameStack | tuple[int, int, int]"
) -> tuple[int, int, int]:
    """
    Returns the shape of the CNN input tensor.

    Parameters:
        frame_stack: FrameStack or tuple[int, int, int]
            Either a FrameStack instance or a shape tuple (channels, height, width).

    Returns:
        tuple[int, int, int]: (channels * stack_size, height, width) if FrameStack,
                              or the tuple itself if already a shape.

    Raises:
        ValueError: If input is not a FrameStack or a tuple of 3 integers.
    """
    if isinstance(frame_stack, FrameStack):
        return (frame_stack.channels * frame_stack.stack_size, frame_stack.height, frame_stack.width)
    if (
        isinstance(frame_stack, tuple)
        and len(frame_stack) == 3
        and all(isinstance(x, int) for x in frame_stack)
    ):
        return frame_stack
    raise ValueError(
        f"frame_stack must be a FrameStack or a tuple of 3 integers (C, H, W), got {type(frame_stack)}: {frame_stack}"
    )


# =========================
# Example Usage (for testing)
# =========================
if __name__ == "__main__":
    import logging

    # Setup a simple logger with log_scheduler_step method for demonstration
    class TestLogger:
        def info(self, msg: str) -> None:
            print(f"[INFO] {msg}")

        def error(self, msg: str) -> None:
            print(f"[ERROR] {msg}")

        def log_scheduler_step(self, msg: str, step: int) -> None:
            print(f"[STEP {step}] {msg}")

    logger = TestLogger()

    # Deterministic sample frame for local module diagnostics.
    y = np.linspace(0, 255, SCREEN_HEIGHT, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, SCREEN_WIDTH, dtype=np.uint8)[None, :]
    sample_frame = np.stack(
        (
            np.broadcast_to(x, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            np.broadcast_to(y, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            np.full((SCREEN_HEIGHT, SCREEN_WIDTH), 96, dtype=np.uint8),
        ),
        axis=2,
    )

    # Initialize frame stack with logger and dynamic channels and stack size
    fs = FrameStack(stack_size=FRAME_STACK, channels=NUM_CHANNELS, logger=logger)
    fs.add_frame(sample_frame, from_neoforge=True, log_scheduler_step=1)

    stacked = fs.get_stack()
    print("Stacked frames shape:", stacked.shape)

    # Initialize CNN
    input_shape = get_cnn_input_shape(fs)
    cnn = CNNFeatureExtractor(input_shape).to(DEVICE)

    # Convert stacked frames to tensor
    tensor_input = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Forward pass
    features = cnn(tensor_input)
    print("Extracted features shape:", features.shape)
