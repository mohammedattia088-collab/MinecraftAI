import asyncio
from OtherStuff.MinecraftAI.environment.minecraft_env import MinecraftEnv
from OtherStuff.MinecraftAI.perception.visual import FrameStack, CNNFeatureExtractor, get_cnn_input_shape, preprocess_frame
from OtherStuff.MinecraftAI.utils.logger import Logger
import torch


async def test_env_step():

    env = MinecraftEnv()
    obs = await env.reset()
    print("Initial frame shape:", obs.shape)

    action_name = list(env.keybinds.keys())[0]  # Press first key
    next_obs, reward, done = await env.step(action_name)
    print("Next frame shape:", next_obs.shape, "Reward:", reward, "Done:", done)

asyncio.run(test_env_step())