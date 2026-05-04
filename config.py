"""
config.py

Centralized configuration file for MinecraftAI project.
Contains all hyperparameters, keybindings, screen/input settings, RL parameters,
reward weights, training parameters, and paths.

Author: MinecraftAI Senior Dev Team
Version: 1.0
"""

import os

# =========================
# Paths & File Locations
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# Screen / Input Settings
# =========================
# Minecraft window capture resolution
SCREEN_WIDTH = 1920   # Adjust if running smaller/larger window
SCREEN_HEIGHT = 1080
NUM_CHANNELS = 3      # RGB
FRAME_STACK = 4       # Number of stacked frames as input to CNN

# =========================
# Key Bindings
# =========================
# Map action names to Minecraft keys (pyautogui / OS events compatible)
KEY_BINDINGS = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "jump": "space",
    "crouch": "shift",
    "attack": "mouse1",
    "use": "mouse2",
    "inventory": "e",
    "hotbar_1": "1",
    "hotbar_2": "2",
    "hotbar_3": "3",
    "hotbar_4": "4",
    "hotbar_5": "5",
    "hotbar_6": "6",
    "hotbar_7": "7",
    "hotbar_8": "8",
    "hotbar_9": "9",
    "offhand": "f",
    "sprint": "ctrl"
}

# =========================
# Reinforcement Learning Hyperparameters
# =========================
# DQN / Double DQN / Dueling Head
RL_PARAMS = {
    "gamma": 0.99,                 # Discount factor
    "learning_rate": 1e-4,         # Adam optimizer
    "batch_size": 64,              # Mini-batch size
    "buffer_size": 100000,         # Experience replay buffer size
    "target_update_freq": 1000,    # Steps between target network updates
    "dueling": True,               # Enable Dueling DQN
    "double_dqn": True,            # Enable Double DQN
    "per_alpha": 0.6,              # Prioritized Experience Replay exponent
    "per_beta_start": 0.4,         # Initial beta for importance sampling
    "per_beta_frames": 1000000     # Frames over which beta is annealed to 1.0
}

# =========================
# Training / Exploration Parameters
# =========================
TRAINING_PARAMS = {
    "max_episodes": 50000,          # Max episodes to train
    "max_steps_per_episode": 1000,  # Max steps per episode
    "epsilon_start": 1.0,           # Initial exploration
    "epsilon_end": 0.05,            # Minimum exploration
    "epsilon_decay": 1e-5,          # Linear decay per step
    "save_every_episodes": 100,     # Checkpoint frequency
    "log_every_episodes": 10        # Logging frequency
}

# =========================
# Reward Weights
# =========================
# Positive rewards for beneficial actions, negative for harmful
REWARD_WEIGHTS = {
    "survival": 1.0,        # Staying alive
    "mine_valuable": 2.0,   # Mining diamonds, gold, etc.
    "craft_item": 1.5,      # Crafting tools, armor, blocks
    "exploration": 0.5,     # Discovering new areas
    "damage_taken": -1.0,   # Losing health
    "fall_death": -2.0,     # Falling from heights
    "lava_death": -3.0,     # Death by lava
    "attack_enemy": 1.0,    # Killing mobs
}

# =========================
# Miscellaneous Settings
# =========================
MPS_DEVICE = True              # Use Metal GPU acceleration on Mac M1
SEED = 42                      # For reproducibility
VERBOSE = True                 # Debug prints during training/inference

# =========================
# Minecraft Bridge Settings
# =========================
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 25575
BRIDGE_AUTH_TOKEN = ""

# =========================
# Autonomous Player Settings
# =========================
AUTONOMOUS_PARAMS = {
    "policy_path": os.path.join(DATA_DIR, "autonomous_policy.json"),
    "experience_log": os.path.join(DATA_DIR, "autonomous_experience.jsonl"),
    "default_goal": "survive and progress",
    "default_steps": 1000,
    "save_every_steps": 25,
    "step_delay": 0.15,
    "request_timeout": 5.0,
    "max_reconnect_delay": 12.0,
    "tactical_override_epsilon": 0.04,
    "goal_override_epsilon": 0.12,
    "utility_policy_mix_epsilon": 0.04,
    "world_memory_positions": 2048,
    "diamond_target_y": -54,
    "critical_health": 6.0,
    "low_health": 10.0,
    "hungry_food": 14,
    "starving_food": 7,
    "combat_radius": 10.0,
    "flee_radius": 18.0,
    "resource_scan_radius": 32,
    "stuck_window": 8,
    "stuck_distance": 0.8,
    "position_history": 24,
    "max_action_failures": 4,
    "experience_flush_every": 10,
    "epsilon_start": 0.35,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9995,
    "learning_rate": 0.18,
    "discount": 0.92,
}

# =========================
# Helper Functions
# =========================
def get_full_path(directory, filename):
    """
    Returns absolute path for a file in a given directory.
    """
    return os.path.join(directory, filename)
