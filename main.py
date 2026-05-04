"""
main.py

Entry point for MinecraftAI project.
Integrates perception, environment, RL agent, logger, and schedulers.
Runs training or inference loops with checkpoints and metrics.

Author: MinecraftAI Senior Dev Team
Version: 1.0
"""

import os
import numpy as np
import torch
import mss
import asyncio

try:
    from .config import CHECKPOINT_DIR, TRAINING_PARAMS, RL_PARAMS, VERBOSE, SEED, SCREEN_WIDTH, SCREEN_HEIGHT, KEY_BINDINGS
    from .perception.visual import FrameStack, get_cnn_input_shape, CNNFeatureExtractor
    from .environment.minecraft_env import MinecraftEnv
    from .agent.rl_agent import RLAgent
    from .utils.logger import Logger
    from .utils.scheduler import EpsilonScheduler, LRScheduler
except ImportError:
    from config import CHECKPOINT_DIR, TRAINING_PARAMS, RL_PARAMS, VERBOSE, SEED, SCREEN_WIDTH, SCREEN_HEIGHT, KEY_BINDINGS
    from perception.visual import FrameStack, get_cnn_input_shape, CNNFeatureExtractor
    from environment.minecraft_env import MinecraftEnv
    from agent.rl_agent import RLAgent
    from utils.logger import Logger
    from utils.scheduler import EpsilonScheduler, LRScheduler

# =========================
# Set random seeds
# =========================
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Initialize Modules
# =========================
# Logger
logger = Logger("MinecraftAI")

# Environment
env = MinecraftEnv()

# Perception (frame stack) with logger
frame_stack = FrameStack(frame_logger=logger)

# CNN input shape
cnn_input_shape = get_cnn_input_shape(frame_stack)
cnn = CNNFeatureExtractor(cnn_input_shape)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cnn.to(DEVICE)

# Dynamically compute state_dim with a zero-valued shape probe.
probe_input = torch.zeros((1, *cnn_input_shape), dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    probe_output = cnn(probe_input)
state_dim = probe_output.shape[1]

# RL Agent
keybinds_list = KEY_BINDINGS if isinstance(KEY_BINDINGS, list) else list(KEY_BINDINGS)
n_actions = len(keybinds_list)

# Epsilon scheduler with type hint
eps_scheduler: EpsilonScheduler = EpsilonScheduler(start=TRAINING_PARAMS['epsilon_start'],
                                 end=TRAINING_PARAMS['epsilon_end'],
                                 decay=TRAINING_PARAMS['epsilon_decay'],
                                 mode='linear')

# Initialize agent first to get optimizer, pass logger as frame_logger
agent = RLAgent(state_dim=state_dim, n_actions=n_actions, use_per=True, frame_logger=logger)

# LR Scheduler with type hint
lr_scheduler: LRScheduler = LRScheduler(optimizer=agent.optimizer,
                                        initial_lr=TRAINING_PARAMS['learning_rate'],
                                        logger=logger)

# Now pass schedulers to agent
agent.set_schedulers(eps_scheduler=eps_scheduler, lr_scheduler=lr_scheduler)

# =========================
# Training Loop
# =========================

async def capture_minecraft_frame():
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
        img = np.array(sct.grab(monitor))
        img = img[:, :, :3]  # Drop alpha channel
        return img

async def train(max_episodes=TRAINING_PARAMS['max_episodes']):
    for episode in range(max_episodes):
        # Reset environment
        obs = await env.reset()
        frame_stack = FrameStack(frame_logger=logger)  # Reset frame stack with logger
        total_reward = 0
        done = False
        step_count = 0

        # Initialize first frame by capturing from environment observation
        frame_stack.add_frame(obs)
        logger.log_metrics({"episode_start": True}, episode)

        while not done and step_count < TRAINING_PARAMS['max_steps_per_episode']:
            # Get stacked frames
            stacked_frames = frame_stack.get_stack()
            stacked_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # Add batch dim and move to device
            with torch.no_grad():
                features = cnn(stacked_tensor).detach().cpu().numpy()[0]

            # Select action
            action_idx = await agent.select_action(features)
            action_name = keybinds_list[action_idx]

            # Step environment
            next_obs, reward, done = await env.step(action_name)
            total_reward += reward
            logger.log_metrics({"step_reward": reward, "action": action_name}, episode, step_count)

            # Preprocess and add the next observed frame.
            frame_stack.add_frame(next_obs)
            next_stacked_frames = frame_stack.get_stack()
            next_stacked_tensor = torch.tensor(next_stacked_frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                next_features = cnn(next_stacked_tensor).detach().cpu().numpy()[0]

            # Store experience
            await agent.store_experience(features, action_idx, reward, next_features, done)

            # Update agent (epsilon and lr schedulers are handled internally)
            await agent.update_model(batch_size=RL_PARAMS['batch_size'])

            step_count += 1

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # Log episode metrics
        logger.log_metrics({
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "steps": step_count
        }, episode)

        # Save checkpoint
        if episode % TRAINING_PARAMS['save_every_episodes'] == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"agent_ep{episode}.pth")
            checkpoint = {
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'eps_scheduler_state_dict': eps_scheduler.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'episode': episode
            }
            torch.save(checkpoint, checkpoint_path)
            if VERBOSE:
                print(f"[INFO] Saved checkpoint: {checkpoint_path}")

        if VERBOSE:
            print(f"[INFO] Episode {episode} finished | Total Reward: {total_reward} | Steps: {step_count}")

# =========================
# Run Training
# =========================
if __name__ == "__main__":
    try:
        asyncio.run(train())
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
    finally:
        logger.close()
        print("[INFO] Logger closed and training stopped.")
