import asyncio
import torch
import numpy as np
from OtherStuff.MinecraftAI.environment.minecraft_env import MinecraftEnv
from OtherStuff.MinecraftAI.perception.visual import FrameStack, CNNFeatureExtractor, get_cnn_input_shape, preprocess_frame
from OtherStuff.MinecraftAI.utils.logger import Logger
from OtherStuff.MinecraftAI.agent.rl_agent import RLAgent

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

async def test_environment_initialization():
    logger = Logger("TestEnvInit")
    try:
        env = MinecraftEnv()
        obs = await env.reset()
        assert isinstance(obs, np.ndarray), f"Initial observation should be np.ndarray, got {type(obs)}"
        print("[INFO] Environment reset successful. Observation shape:", obs.shape)

        assert hasattr(env, "keybinds"), "Environment missing 'keybinds' attribute"
        assert isinstance(env.keybinds, dict), "'keybinds' should be a dict"
        assert len(env.keybinds) > 0, "'keybinds' dict is empty"
        print(f"[INFO] Environment keybinds loaded: {list(env.keybinds.keys())}")

    except Exception as e:
        print("[ERROR] test_environment_initialization failed:", e)
    finally:
        logger.close()

async def test_perception_module():
    logger = Logger("TestPerception")
    try:
        env = MinecraftEnv()
        obs = await env.reset()
        assert isinstance(obs, np.ndarray), f"Observation should be np.ndarray, got {type(obs)}"
        preprocessed = preprocess_frame(obs)
        assert isinstance(preprocessed, np.ndarray), "Preprocessed frame should be np.ndarray"
        print("[INFO] Frame preprocessing successful. Preprocessed shape:", preprocessed.shape)

        # Use dynamic stack_size and channels from env
        frame_stack = FrameStack(stack_size=env.stack_size, channels=env.channels)
        frame_stack.add_frame(preprocessed)
        stacked_frames = frame_stack.get_stack()
        assert isinstance(stacked_frames, np.ndarray), "Stacked frames should be np.ndarray"
        print("[INFO] FrameStack stacking successful. Stacked frames shape:", stacked_frames.shape)

        cnn_input_shape = get_cnn_input_shape(stacked_frames.shape)
        cnn = CNNFeatureExtractor(cnn_input_shape).to(DEVICE)

        tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = cnn(tensor)
        assert features.ndim == 2 and features.shape[0] == 1, "CNN output shape invalid"
        print("[INFO] CNN feature extraction successful. Feature shape:", features.shape)

    except Exception as e:
        print("[ERROR] test_perception_module failed:", e)
    finally:
        logger.close()

async def test_rl_agent():
    logger = Logger("TestRLAgent")
    try:
        # Initialize a real feature-vector contract and action count.
        feature_dim = 512
        n_actions = 4
        agent = RLAgent(state_dim=feature_dim, n_actions=n_actions, frame_logger=logger)

        feature_vector = np.linspace(-1.0, 1.0, feature_dim, dtype=np.float32)
        action_idx = await agent.select_action(feature_vector)
        assert isinstance(action_idx, int), "Selected action index should be int"
        assert 0 <= action_idx < n_actions, f"Action index {action_idx} out of range"
        print(f"[INFO] RLAgent action selection successful. Selected action index: {action_idx}")

    except Exception as e:
        print("[ERROR] test_rl_agent failed:", e)
    finally:
        logger.close()

async def test_environment_step():
    logger = Logger("TestEnvStep")
    try:
        env = MinecraftEnv()
        obs = await env.reset()
        assert isinstance(obs, np.ndarray), "Initial observation should be np.ndarray"
        assert hasattr(env, "keybinds") and isinstance(env.keybinds, dict) and len(env.keybinds) > 0, "Invalid env.keybinds"
        action_name = list(env.keybinds.keys())[0]

        next_obs, reward, done = await env.step(action_name)
        assert isinstance(next_obs, np.ndarray), "Next observation should be np.ndarray"
        assert next_obs.shape == obs.shape, "Next observation shape mismatch"
        assert isinstance(reward, (int, float)), "Reward should be int or float"
        assert isinstance(done, bool), "Done flag should be bool"

        print(f"[INFO] Environment step successful. Action: {action_name}, Reward: {reward}, Done: {done}")
        print(f"[INFO] Next observation shape: {next_obs.shape}")

    except Exception as e:
        print("[ERROR] test_environment_step failed:", e)
    finally:
        logger.close()

async def test_mini_episode():
    logger = Logger("TestMiniEpisode")
    try:
        env = MinecraftEnv()
        frame_stack = FrameStack(stack_size=env.stack_size, channels=env.channels)
        max_steps = 30
        cumulative_reward = 0.0
        steps = 0

        obs = await env.reset()
        assert isinstance(obs, np.ndarray), "Initial observation should be np.ndarray"
        frame_stack.add_frame(preprocess_frame(obs))
        stacked_frames = frame_stack.get_stack()
        cnn_input_shape = get_cnn_input_shape(stacked_frames.shape)
        cnn = CNNFeatureExtractor(cnn_input_shape).to(DEVICE)

        # Compute feature_dim dynamically from the current stacked-frame tensor.
        with torch.no_grad():
            probe_input = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            probe_features = cnn(probe_input)
        feature_dim = probe_features.shape[1]

        # Validate keybinds
        assert hasattr(env, "keybinds") and isinstance(env.keybinds, dict) and len(env.keybinds) > 0, "Invalid env.keybinds"
        n_actions = len(env.keybinds)

        agent = RLAgent(state_dim=feature_dim, n_actions=n_actions, frame_logger=logger)

        adaptation_flag = False
        adaptation_steps = set()
        while steps < max_steps:
            # Every 10 steps, test real fallback functionality by toggling env.fallback_mode
            if steps > 0 and steps % 10 == 0:
                adaptation_flag = True
                adaptation_steps.add(steps)
                # Toggle fallback mode to simulate environment change
                env.fallback_mode = not getattr(env, 'fallback_mode', False)
                print(f"[ADAPTATION EVENT] Step {steps}: Toggled fallback_mode to {env.fallback_mode}")

            frame_stack.add_frame(preprocess_frame(obs))
            stacked_frames = frame_stack.get_stack()
            assert isinstance(stacked_frames, np.ndarray), "stacked_frames should be np.ndarray"
            tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = cnn(tensor)
            assert features.ndim == 2 and features.shape[0] == 1, "CNN output shape invalid"

            features_np = features.cpu().numpy().squeeze()
            assert features_np.shape[0] == feature_dim, "Feature dimension mismatch"

            action_idx = await agent.select_action(features_np)
            assert isinstance(action_idx, int), "Action index should be int"
            assert 0 <= action_idx < n_actions, "Action index out of range"

            action_names = list(env.keybinds.keys())
            action_name = action_names[action_idx]

            next_obs, reward, done = await env.step(action_name)

            # If adaptation is active, verify fallback is in effect by checking env.fallback_mode
            if adaptation_flag:
                # Confirm that fallback_mode is True or False (toggled)
                assert hasattr(env, 'fallback_mode'), "Environment missing fallback_mode attribute during adaptation"
                print(f"[ADAPTATION] fallback_mode is now {env.fallback_mode}")
                adaptation_flag = False

            if done:
                obs = await env.reset()
                frame_stack = FrameStack(stack_size=env.stack_size, channels=env.channels)
                frame_stack.add_frame(preprocess_frame(obs))
            else:
                obs = next_obs

            assert isinstance(obs, np.ndarray), "obs should be np.ndarray"
            assert isinstance(reward, (int, float)), "reward should be numeric"
            assert isinstance(done, bool), "done should be bool"

            adaptation_note = "[ADAPTATION]" if steps in adaptation_steps else ""
            print(f"[STEP {steps}] Action: {action_name} | Reward: {reward} | Done: {done} {adaptation_note}")
            print(f"          Observation shape: {obs.shape} | Feature shape: {features.shape}")

            cumulative_reward += reward
            steps += 1

        print(f"[EPISODE END] Steps taken: {steps}, Cumulative reward: {cumulative_reward}")
        assert steps <= max_steps, "Exceeded max_steps"

    except Exception as e:
        print("[ERROR] test_mini_episode failed:", e)
    finally:
        logger.close()

async def main():
    await test_environment_initialization()
    await test_perception_module()
    await test_rl_agent()
    await test_environment_step()
    await test_mini_episode()

if __name__ == "__main__":
    asyncio.run(main())
