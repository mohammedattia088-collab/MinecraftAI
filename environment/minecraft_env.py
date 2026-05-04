"""
Bridge-backed Minecraft environment for MinecraftAI.

The environment keeps the existing visual-observation API for the DQN code while
using the Minecraft Bridge mod as the authoritative source for actions, health,
position, inventory, threats, and reward shaping.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional local dependency
    cv2 = None

try:
    from PIL import ImageGrab
except Exception:  # pragma: no cover - optional local dependency
    ImageGrab = None

try:
    import mss
except Exception:  # pragma: no cover - optional local dependency
    mss = None

try:
    from pynput.keyboard import Controller as KeyboardController, Key
    from pynput.mouse import Controller as MouseController, Button
except Exception:  # pragma: no cover - optional local dependency
    KeyboardController = None
    MouseController = None
    Key = None
    Button = None

try:
    from ..bridge import BridgeClient, BridgeError
    from ..config import (
        BRIDGE_AUTH_TOKEN,
        BRIDGE_HOST,
        BRIDGE_PORT,
        KEY_BINDINGS,
        REWARD_WEIGHTS,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
    )
    from ..perception.visual import VisionMemory
except ImportError:
    try:
        from bridge import BridgeClient, BridgeError
        from config import (
            BRIDGE_AUTH_TOKEN,
            BRIDGE_HOST,
            BRIDGE_PORT,
            KEY_BINDINGS,
            REWARD_WEIGHTS,
            SCREEN_HEIGHT,
            SCREEN_WIDTH,
        )
        from perception.visual import VisionMemory
    except ImportError:
        from OtherStuff.MinecraftAI.bridge import BridgeClient, BridgeError
        from OtherStuff.MinecraftAI.config import (
            BRIDGE_AUTH_TOKEN,
            BRIDGE_HOST,
            BRIDGE_PORT,
            KEY_BINDINGS,
            REWARD_WEIGHTS,
            SCREEN_HEIGHT,
            SCREEN_WIDTH,
        )
        from OtherStuff.MinecraftAI.perception.visual import VisionMemory


logger = logging.getLogger("MinecraftEnv")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(handler)


keyboard = KeyboardController() if KeyboardController is not None else None
mouse = MouseController() if MouseController is not None else None


def blank_frame() -> np.ndarray:
    return np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)


def capture_minecraft_frame() -> np.ndarray:
    if mss is not None:
        try:
            monitor = {"left": 0, "top": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
            with mss.mss() as screen_capture:
                shot = np.array(screen_capture.grab(monitor), dtype=np.uint8)
            return shot[:, :, :3][:, :, ::-1].copy()
        except Exception as exc:
            logger.debug("mss screen capture unavailable: %s", exc)

    if ImageGrab is None:
        return blank_frame()
    try:
        bbox = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        return np.array(ImageGrab.grab(bbox).convert("RGB"))
    except Exception as exc:
        logger.debug("Screen capture unavailable: %s", exc)
        return blank_frame()


def extract_health_from_frame(frame: np.ndarray) -> int:
    if cv2 is None:
        return 20
    try:
        heart_region = frame[SCREEN_HEIGHT - 50:SCREEN_HEIGHT - 30, 20:220]
        hsv = cv2.cvtColor(heart_region, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2),
        )
        hearts = max(0, min(int(cv2.countNonZero(mask) / 18), 10))
        return hearts * 2
    except Exception:
        return 20


def detect_danger_from_frame(frame: np.ndarray) -> bool:
    if cv2 is None:
        return False
    try:
        lava_region = frame[
            SCREEN_HEIGHT - 60:SCREEN_HEIGHT - 10,
            SCREEN_WIDTH // 2 - 50:SCREEN_WIDTH // 2 + 50,
        ]
        hsv = cv2.cvtColor(lava_region, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([5, 150, 150]), np.array([15, 255, 255]))
        return cv2.countNonZero(mask) > 500
    except Exception:
        return False


class BridgeMinecraftInterface:
    def __init__(
        self,
        host: str = BRIDGE_HOST,
        port: int = BRIDGE_PORT,
        auth_token: str = BRIDGE_AUTH_TOKEN,
        timeout: float = 5.0,
    ) -> None:
        self.client = BridgeClient(host=host, port=port, auth_token=auth_token, timeout=timeout)
        self.available = False

    async def connect(self) -> bool:
        try:
            await asyncio.wait_for(self.client.connect(), timeout=self.client.timeout)
            self.available = True
            return True
        except (OSError, BridgeError, asyncio.TimeoutError):
            self.available = False
            return False

    async def close(self) -> None:
        await self.client.close()
        self.available = False

    async def request(self, action: str, **payload: Any) -> Dict[str, Any]:
        if not self.client.connected:
            connected = await self.connect()
            if not connected:
                raise BridgeError("Bridge is not available")
        self.available = True
        return await asyncio.wait_for(self.client.request(action, **payload), timeout=self.client.timeout + 1.0)

    async def full_state(self) -> Optional[Dict[str, Any]]:
        try:
            return await self.request("get_full_state")
        except (BridgeError, OSError, asyncio.TimeoutError):
            self.available = False
            return None

    async def execute_action(self, action: str, action_delay: float) -> bool:
        duration = max(0.12, action_delay * 1.5)
        try:
            if action == "forward":
                await self.request("move", forward=1.0, strafe=0.0, duration=duration)
            elif action == "backward":
                await self.request("move", forward=-0.7, strafe=0.0, duration=duration)
            elif action == "left":
                await self.request("move", forward=0.0, strafe=1.0, duration=duration)
            elif action == "right":
                await self.request("move", forward=0.0, strafe=-1.0, duration=duration)
            elif action == "jump":
                await self.request("jump")
            elif action == "crouch":
                await self.request("sneak", state=True)
                await asyncio.sleep(duration)
                await self.request("sneak", state=False)
            elif action == "sprint":
                await self.request("sprint", state=True)
                await self.request("move", forward=1.0, strafe=0.0, duration=duration)
            elif action == "attack":
                await self.request("auto_attack", radius=8.0, approach=True, select_weapon=True)
            elif action == "use":
                await self.request("use")
            elif action.startswith("hotbar_"):
                await self.request("select_slot", slot=max(0, min(8, int(action.rsplit("_", 1)[1]) - 1)))
            else:
                return False
            return True
        except (BridgeError, OSError, asyncio.TimeoutError) as exc:
            logger.debug("Bridge action failed for %s: %s", action, exc)
            self.available = False
            return False


class MinecraftEnv:
    def __init__(
        self,
        keybinds: Dict[str, str] = KEY_BINDINGS,
        safe_mode: bool = True,
        action_delay: float = 0.05,
        stack_size: int = 4,
        channels: int = 3,
        use_bridge: bool = True,
        bridge: Optional[BridgeMinecraftInterface] = None,
    ) -> None:
        self.keybinds = keybinds
        self.safe_mode = safe_mode
        self.action_delay = action_delay
        self.stack_size = stack_size
        self.channels = channels
        self.bridge = bridge or BridgeMinecraftInterface()
        self.use_bridge = use_bridge
        self.alive = True
        self.health = 20
        self.last_health = 20
        self.last_position = (0.0, 0.0, 0.0)
        self.inventory: Dict[str, Any] = {}
        self.last_frame: Optional[np.ndarray] = None
        self.last_bridge_state: Optional[Dict[str, Any]] = None
        self.vision_memory = VisionMemory()
        self.last_visual_analysis: Dict[str, Any] = {}
        logger.info("MinecraftEnv initialized with bridge-backed state and action control.")

    async def step(self, action: str) -> Tuple[np.ndarray, float, bool]:
        if not self.alive:
            return blank_frame(), 0.0, True

        handled = False
        if self.use_bridge:
            if not self.bridge.available:
                await self.bridge.connect()
            handled = await self.bridge.execute_action(action, self.action_delay)

        if not handled:
            await self._press_key(action)

        await asyncio.sleep(self.action_delay)
        frame = await self._get_state()
        reward = await self._compute_reward(frame)
        self.check_safety(frame)
        return frame, reward, not self.alive

    async def _press_key(self, action: str) -> None:
        key = self.keybinds.get(action)
        if key is None or keyboard is None:
            return

        try:
            if key == "mouse1" and mouse is not None and Button is not None:
                mouse.press(Button.left)
                mouse.release(Button.left)
            elif key == "mouse2" and mouse is not None and Button is not None:
                mouse.press(Button.right)
                mouse.release(Button.right)
            elif len(key) == 1:
                keyboard.press(key)
                keyboard.release(key)
            elif Key is not None:
                special_key = getattr(Key, key, None)
                if special_key is not None:
                    keyboard.press(special_key)
                    keyboard.release(special_key)
        except Exception as exc:
            logger.debug("Local input failed for %s: %s", action, exc)

    async def _compute_reward(self, frame: Optional[np.ndarray]) -> float:
        reward = 0.0
        state = await self.bridge.full_state() if self.use_bridge else None
        if state is not None:
            self.last_bridge_state = state
            reward += self._bridge_reward(state)
        elif frame is not None:
            current_health = extract_health_from_frame(frame)
            damage = max(0, self.last_health - current_health)
            self.health = current_health
            self.alive = current_health > 0
            reward += REWARD_WEIGHTS.get("survival", 1.0) if self.alive else -REWARD_WEIGHTS.get("fall_death", 2.0)
            reward += REWARD_WEIGHTS.get("damage_taken", -1.0) * damage
            self.last_health = current_health
        return reward

    def _bridge_reward(self, state: Dict[str, Any]) -> float:
        player = state.get("player", {})
        inventory = state.get("inventory", {})
        threats = state.get("threats", {})

        current_health = float(player.get("health", self.health))
        damage = max(0.0, float(self.last_health) - current_health)
        self.health = int(max(0.0, min(20.0, current_health)))
        self.alive = current_health > 0.0

        reward = REWARD_WEIGHTS.get("survival", 1.0) if self.alive else -REWARD_WEIGHTS.get("fall_death", 2.0)
        reward += REWARD_WEIGHTS.get("damage_taken", -1.0) * damage

        current_position = self._position_from_player(player)
        distance = float(np.linalg.norm(np.array(current_position) - np.array(self.last_position)))
        reward += REWARD_WEIGHTS.get("exploration", 0.1) * min(distance, 12.0)
        self.last_position = current_position

        current_counts = self._inventory_counts(inventory)
        previous_counts = self._inventory_counts(self.inventory)
        total_delta = current_counts["total"] - previous_counts["total"]
        valuable_delta = current_counts["valuable"] - previous_counts["valuable"]
        reward += max(0, total_delta) * 0.05
        reward += max(0, valuable_delta) * REWARD_WEIGHTS.get("mine_valuable", 2.0)
        self.inventory = inventory

        visual = self.last_visual_analysis or {}
        rolling = visual.get("rolling", {}) if isinstance(visual, dict) else {}
        reward += float(rolling.get("novelty", visual.get("novelty_score", 0.0))) * 0.03
        if visual.get("needs_attention"):
            reward -= 0.04

        danger = str(threats.get("danger_level", "none"))
        if player.get("is_in_lava") or player.get("is_on_fire") or danger in {"critical", "high"}:
            reward -= 3.0
        if float(player.get("fall_distance", 0.0)) > 6.0:
            reward -= 1.0

        self.last_health = self.health
        return float(reward)

    async def _get_state(self) -> np.ndarray:
        frame = capture_minecraft_frame()
        self.last_frame = frame.copy()
        try:
            self.last_visual_analysis = self.vision_memory.add_frame(frame)
        except Exception as exc:
            logger.debug("Visual analysis failed: %s", exc)
            self.last_visual_analysis = {}
        return frame

    async def reset(self) -> np.ndarray:
        self.alive = True
        self.health = 20
        self.last_health = 20
        self.last_position = (0.0, 0.0, 0.0)
        self.inventory = {}
        self.last_bridge_state = None
        self.last_frame = None
        self.vision_memory = VisionMemory()
        self.last_visual_analysis = {}

        if self.use_bridge:
            connected = await self.bridge.connect()
            if connected:
                await self._safe_bridge_request("set_unpause", state=True)
                await self._safe_bridge_request("stop_moving")
                state = await self.bridge.full_state()
                if state is not None:
                    self.last_bridge_state = state
                    player = state.get("player", {})
                    self.health = int(float(player.get("health", 20)))
                    self.last_health = self.health
                    self.last_position = self._position_from_player(player)
                    self.inventory = state.get("inventory", {})

        return await self._get_state()

    async def close(self) -> None:
        await self.bridge.close()

    async def _safe_bridge_request(self, action: str, **payload: Any) -> Optional[Dict[str, Any]]:
        try:
            return await self.bridge.request(action, **payload)
        except (BridgeError, OSError, asyncio.TimeoutError):
            return None

    def check_safety(self, frame: Optional[np.ndarray] = None) -> None:
        if not self.safe_mode:
            return

        state = self.last_bridge_state
        if state is not None:
            player = state.get("player", {})
            threats = state.get("threats", {})
            danger = str(threats.get("danger_level", "none"))
            if player.get("is_in_lava") or player.get("is_on_fire") or danger == "critical":
                self.alive = False
            return

        if frame is not None and detect_danger_from_frame(frame):
            self.alive = False

    @staticmethod
    def _position_from_player(player: Dict[str, Any]) -> Tuple[float, float, float]:
        position = player.get("position", {})
        return (
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        )

    @staticmethod
    def _inventory_counts(inventory: Dict[str, Any]) -> Dict[str, int]:
        valuable_terms = ("diamond", "emerald", "ancient_debris", "gold", "iron", "redstone", "lapis")
        total = 0
        valuable = 0
        for item in inventory.get("items", []):
            name = str(item.get("item", ""))
            count = int(item.get("count", 0))
            total += count
            if any(term in name for term in valuable_terms):
                valuable += count
        return {"total": total, "valuable": valuable}


if __name__ == "__main__":
    async def run_env() -> None:
        env = MinecraftEnv()
        await env.reset()
        for index in range(10):
            _, reward, done = await env.step("forward")
            logger.info("Step %s: reward=%s done=%s", index, reward, done)
            if done:
                break
        await env.close()

    asyncio.run(run_env())
