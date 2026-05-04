import asyncio
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

try:
    from ..bridge import BridgeClient, BridgeError
    from ..config import AUTONOMOUS_PARAMS, BRIDGE_AUTH_TOKEN, BRIDGE_HOST, BRIDGE_PORT
    from ..mechanics import BasicMechanics, SkillResult
    from ..perception import WorldMemory, build_world_state
    from ..planning import ObjectivePlanner, UtilityPlanner
except ImportError:
    from bridge import BridgeClient, BridgeError
    from config import AUTONOMOUS_PARAMS, BRIDGE_AUTH_TOKEN, BRIDGE_HOST, BRIDGE_PORT
    from mechanics import BasicMechanics, SkillResult
    from perception import WorldMemory, build_world_state
    from planning import ObjectivePlanner, UtilityPlanner


WOOD_TERMS = ("log", "wood", "planks", "sapling", "stick", "crafting_table")
TOOL_TERMS = ("pickaxe", "axe", "shovel", "sword", "hoe")
ARMOR_TERMS = ("helmet", "chestplate", "leggings", "boots", "shield")
RESOURCE_TERMS = ("log", "ore", "coal", "iron", "copper", "gold", "diamond", "redstone", "lapis", "emerald")
DIAMOND_TERMS = ("diamond", "diamond_ore", "deepslate_diamond_ore")


@dataclass
class QPolicy:
    actions: Tuple[str, ...]
    path: str
    alpha: float = 0.18
    gamma: float = 0.92
    epsilon: float = 0.35
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    steps: int = 0

    @classmethod
    def load(cls, path: str, actions: Tuple[str, ...]) -> "QPolicy":
        params = AUTONOMOUS_PARAMS
        policy = cls(
            actions=actions,
            path=path,
            alpha=float(params["learning_rate"]),
            gamma=float(params["discount"]),
            epsilon=float(params["epsilon_start"]),
            epsilon_min=float(params["epsilon_min"]),
            epsilon_decay=float(params["epsilon_decay"]),
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            policy.table = {
                state: {action: float(value) for action, value in actions_dict.items()}
                for state, actions_dict in raw.get("table", {}).items()
            }
            policy.epsilon = float(raw.get("epsilon", policy.epsilon))
            policy.steps = int(raw.get("steps", 0))
        return policy

    def choose(self, state: str) -> str:
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        values = self.table[state]
        return max(self.actions, key=lambda action: (values.get(action, 0.0), random.random()))

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_value = self.table[state].get(action, 0.0)
        future = max(self.table[next_state].values()) if self.table[next_state] else 0.0
        target = reward + self.gamma * future
        self.table[state][action] = old_value + self.alpha * (target - old_value)
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {
            "version": 1,
            "steps": self.steps,
            "epsilon": self.epsilon,
            "actions": list(self.actions),
            "table": self.table,
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path)

    def _ensure_state(self, state: str) -> None:
        if state not in self.table:
            self.table[state] = {action: 0.0 for action in self.actions}


class AutonomousPlayer:
    """
    Long-running Minecraft learner.

    It combines a utility planner, structured world memory, reusable mechanics,
    and a tabular fallback policy over symbolic bridge observations. The planner
    handles urgent survival and progression; the policy still explores within the
    available skill set and keeps useful experience on disk.
    """

    def __init__(
        self,
        host: str = BRIDGE_HOST,
        port: int = BRIDGE_PORT,
        auth_token: str = BRIDGE_AUTH_TOKEN,
        policy_path: str = AUTONOMOUS_PARAMS["policy_path"],
        experience_log: str = AUTONOMOUS_PARAMS["experience_log"],
        request_timeout: float = AUTONOMOUS_PARAMS["request_timeout"],
        step_delay: float = AUTONOMOUS_PARAMS["step_delay"],
        goal: str = AUTONOMOUS_PARAMS.get("default_goal", "survive and progress"),
    ) -> None:
        self.client = BridgeClient(
            host=host,
            port=port,
            auth_token=auth_token,
            timeout=request_timeout,
            max_reconnect_delay=float(AUTONOMOUS_PARAMS["max_reconnect_delay"]),
        )
        self.mechanics = BasicMechanics(self.client, step_delay=step_delay)
        self.policy = QPolicy.load(policy_path, BasicMechanics.SKILLS)
        self.experience_log = experience_log
        self.save_every_steps = int(AUTONOMOUS_PARAMS["save_every_steps"])
        self.step_delay = step_delay
        self.goal = self.normalize_goal(goal)
        self.objective_planner = ObjectivePlanner(self.goal)
        self.utility_planner = UtilityPlanner(self.goal)
        self.world_memory = WorldMemory(max_positions=int(AUTONOMOUS_PARAMS.get("world_memory_positions", 2048)))
        self.position_history: Deque[Tuple[float, float, float]] = deque(
            maxlen=int(AUTONOMOUS_PARAMS["position_history"])
        )
        self.recent_failures: Deque[bool] = deque(maxlen=int(AUTONOMOUS_PARAMS["max_action_failures"]))
        self._experience_buffer: List[Dict[str, Any]] = []

    async def run(self, steps: Optional[int] = None, forever: bool = False) -> None:
        await self.client.ensure_connected()
        await self._safe_request("set_unpause", state=True)
        observation = await self._observe_with_retry()
        self._remember_position(observation)
        step = 0

        while forever or steps is None or step < steps:
            state = self.state_key(observation)
            action, decision_source, decision_reason = self.choose_action(observation, state)
            started = time.time()
            result = await self.mechanics.run(action, observation)
            await asyncio.sleep(self.step_delay)
            next_observation = await self._observe_with_retry()
            next_state = self.state_key(next_observation)
            reward = self.reward(observation, next_observation, result)
            self.policy.update(state, action, reward, next_state)
            self._remember_position(next_observation)
            self.recent_failures.append(not result.success)
            self.world_memory.record_result(result.name, result.success)
            self._record_experience(
                state,
                action,
                reward,
                next_state,
                result,
                time.time() - started,
                decision_source,
                decision_reason,
            )
            observation = next_observation
            step += 1

            if step % self.save_every_steps == 0:
                self.policy.save()
                self._flush_experience()
                print(
                    f"[autonomous] step={self.policy.steps} epsilon={self.policy.epsilon:.3f} "
                    f"goal={self.goal!r} action={action} source={decision_source} "
                    f"reward={reward:.3f} state={next_state}"
                )

        self.policy.save()
        self._flush_experience()
        await self.mechanics.stop_all()

    async def _safe_request(self, action: str, **payload: Any) -> Optional[Dict[str, Any]]:
        try:
            return await self.client.request(action, **payload)
        except BridgeError as exc:
            print(f"[autonomous] bridge command failed: {action}: {exc}")
            return None

    async def _observe_with_retry(self) -> Dict[str, Any]:
        while True:
            try:
                return await self.mechanics.observe()
            except (BridgeError, OSError):
                await asyncio.sleep(self.client.reconnect_delay)
                await self.client.ensure_connected()

    def choose_action(self, observation: Dict[str, Any], state: str) -> Tuple[str, str, str]:
        world = build_world_state(observation)
        recent_failures = sum(1 for failed in self.recent_failures if failed)
        utility = self.utility_planner.decide(
            world,
            self.world_memory,
            stuck=self.is_stuck(),
            recent_failures=recent_failures,
        )
        self.world_memory.update(world)
        policy_mix_epsilon = float(AUTONOMOUS_PARAMS.get("utility_policy_mix_epsilon", 0.04))
        if utility.score >= 0.25 or random.random() >= policy_mix_epsilon:
            reason = f"{utility.reason}; alternatives={utility.alternatives}"
            return utility.skill, "utility", reason
        reason = f"policy_mix_after_low_utility:{utility.reason}"

        if self.is_stuck() and not self.in_high_danger(observation):
            return random.choice(("look_around", "jump_forward", "scan_environment")), "unstuck", "low_progress"

        if self.too_many_recent_failures():
            return random.choice(("survival_check", "scan_environment", "look_around")), "recovery", "recent_failures"

        return self.policy.choose(state), "policy", reason

    def choose_goal_skill(self, observation: Dict[str, Any]) -> Tuple[Optional[str], str]:
        objective = self.objective_planner.select_skill(observation)
        if objective is not None:
            return objective.skill, f"{objective.objective}:{objective.reason}"

        if not self.goal or self.goal in {"survive", "survive and progress", "progress"}:
            return None, "no_goal"

        if "diamond" in self.goal:
            if self.inventory_item_count(observation.get("inventory", {}), DIAMOND_TERMS) > 0:
                return "scan_environment", "diamond_acquired_keep_safe"
            if self.nearby_block_matches(observation, DIAMOND_TERMS):
                return "mine_nearest_diamond", "diamond_visible"
            y = self.position(observation.get("player", {}))[1]
            diamond_y = float(AUTONOMOUS_PARAMS.get("diamond_target_y", -54))
            if y > diamond_y + 8:
                return "descend_to_diamond_layer", "reach_diamond_layer"
            return random.choice(("mine_nearest_diamond", "branch_mine", "scan_environment")), "branch_search_diamond"

        if "achievement" in self.goal or "advancement" in self.goal:
            inv = self.inventory_counts(observation.get("inventory", {}))
            if inv["wood"] <= 0:
                return random.choice(("scan_logs", "mine_nearest_log", "harvest_trees")), "advancement_get_wood"
            if inv["tools"] <= 0:
                return random.choice(("mine_nearest_resource", "scan_environment", "branch_mine")), "advancement_get_tools"
            if self.nearby_block_matches(observation, RESOURCE_TERMS):
                return "mine_nearest_resource", "advancement_collect_resources"
            if self.closest_hostile_nearby(observation):
                return "engage_hostile", "advancement_combat_practice"
            return random.choice(("sprint_wander", "scan_environment", "mine_nearest_resource")), "advancement_explore"

        if "wood" in self.goal or "tree" in self.goal:
            return random.choice(("scan_logs", "mine_nearest_log", "harvest_trees")), "wood_goal"

        if "explore" in self.goal:
            return random.choice(("sprint_wander", "scan_environment", "look_around")), "explore_goal"

        return None, "unrecognized_goal"

    def state_key(self, observation: Dict[str, Any]) -> str:
        player = observation.get("player", {})
        environment = observation.get("environment", {})
        inventory = observation.get("inventory", {})
        nearby_blocks = observation.get("nearby_blocks", [])
        entities = observation.get("nearby_entities", [])
        threats = observation.get("threats", {})
        survival = observation.get("survival", {})
        combat = observation.get("combat", {})
        vision = observation.get("vision", {})
        objective_bucket = self.objective_planner.state_bucket(observation)

        health = float(player.get("health", 20.0))
        food = float(player.get("food", 20.0))
        y = float(player.get("position", {}).get("y", 64.0))
        air = float(player.get("air", player.get("max_air", 300)))
        max_air = max(1.0, float(player.get("max_air", 300)))

        health_bucket = self.bucket(health, (5, 12, 19), ("critical", "low", "ok", "full"))
        food_bucket = self.bucket(food, (6, 14, 19), ("starving", "hungry", "ok", "full"))
        y_bucket = self.bucket(y, (45, 63, 90), ("deep", "low", "surface", "high"))
        oxygen_bucket = self.bucket(air / max_air, (0.25, 0.6, 0.99), ("air_critical", "air_low", "air_recovering", "air_full"))
        time_bucket = "day" if environment.get("is_day", True) else "night"
        danger = str(threats.get("danger_level", "danger" if player.get("is_in_lava") or player.get("is_on_fire") else "safe"))
        hostile = "hostile" if any(BasicMechanics.is_hostile(entity) for entity in entities) else "calm"
        item_near = "item" if any(entity.get("type") == "minecraft:item" for entity in entities) else "no_item"
        resource_near = "resource" if any(
            bool(block.get("resource")) or any(term in str(block.get("block", "")) for term in RESOURCE_TERMS)
            for block in nearby_blocks
        ) else "no_resource"
        visible_count = int(vision.get("visible_entity_count", 0))
        visible_bucket = self.bucket(visible_count, (1, 4, 10), ("blind", "some_vision", "busy_vision", "crowded"))
        inv = self.inventory_counts(inventory)
        wood_bucket = self.bucket(inv["wood"], (1, 8, 32), ("no_wood", "some_wood", "wood_stack", "wood_rich"))
        tool_bucket = "tools" if inv["tools"] > 0 else "no_tools"
        armor_bucket = "armor" if inv["armor"] > 0 or int(player.get("armor_value", 0)) > 0 else "no_armor"
        combat_bucket = "can_attack" if combat.get("can_attack_now") else "combat_ready" if combat.get("has_target") else "no_target"
        stuck_bucket = "stuck" if self.is_stuck() else "mobile"

        return "|".join((
            health_bucket,
            food_bucket,
            oxygen_bucket,
            y_bucket,
            time_bucket,
            danger,
            hostile,
            item_near,
            resource_near,
            visible_bucket,
            wood_bucket,
            tool_bucket,
            armor_bucket,
            combat_bucket,
            stuck_bucket,
            objective_bucket,
            "eat" if survival.get("should_eat") else "fed",
        ))

    def reward(self, before: Dict[str, Any], after: Dict[str, Any], result: SkillResult) -> float:
        reward = 0.03 + result.reward_hint

        before_player = before.get("player", {})
        after_player = after.get("player", {})
        before_health = float(before_player.get("health", 20.0))
        after_health = float(after_player.get("health", 20.0))
        health_delta = after_health - before_health
        reward += health_delta * 0.4
        if after_health <= 0:
            reward -= 10.0
        if after_player.get("is_in_lava") or after_player.get("is_on_fire"):
            reward -= 4.0
        if BasicMechanics.in_environmental_danger(after):
            reward -= 0.8

        moved = self.distance(self.position(before_player), self.position(after_player))
        reward += min(moved, 8.0) * 0.035
        if self.is_stuck():
            reward -= 0.12

        before_inv = self.inventory_counts(before.get("inventory", {}))
        after_inv = self.inventory_counts(after.get("inventory", {}))
        reward += max(0, after_inv["total"] - before_inv["total"]) * 0.08
        reward += max(0, after_inv["wood"] - before_inv["wood"]) * 0.35
        reward += max(0, after_inv["tools"] - before_inv["tools"]) * 0.6
        reward += max(0, after_inv["armor"] - before_inv["armor"]) * 0.3

        before_threat = float(before.get("threats", {}).get("score", 0.0))
        after_threat = float(after.get("threats", {}).get("score", 0.0))
        reward += max(0.0, before_threat - after_threat) * 0.08
        reward -= max(0.0, after_threat - before_threat) * 0.04

        combat_details = result.details if isinstance(result.details, dict) else {}
        if combat_details.get("attacked"):
            reward += 0.45
        if combat_details.get("approaching"):
            reward += 0.08
        if result.name in {"flee_hostile", "kite_hostile"} and after_threat < before_threat:
            reward += 0.25
        if result.name == "eat_food" and after_health >= before_health:
            reward += 0.15

        reward += self.goal_reward(before, after, result)
        reward += self.objective_planner.progression_reward(before, after, result.name)

        if any(BasicMechanics.is_hostile(entity) and float(entity.get("distance", 999.0)) < 8.0 for entity in after.get("nearby_entities", [])):
            reward -= 0.08
        if not result.success:
            reward -= 0.06
        return reward

    def goal_reward(self, before: Dict[str, Any], after: Dict[str, Any], result: SkillResult) -> float:
        reward = 0.0
        if "diamond" in self.goal:
            before_diamonds = self.inventory_item_count(before.get("inventory", {}), DIAMOND_TERMS)
            after_diamonds = self.inventory_item_count(after.get("inventory", {}), DIAMOND_TERMS)
            reward += max(0, after_diamonds - before_diamonds) * 10.0

            before_y = self.position(before.get("player", {}))[1]
            after_y = self.position(after.get("player", {}))[1]
            target_y = float(AUTONOMOUS_PARAMS.get("diamond_target_y", -54))
            if before_y > target_y and after_y < before_y:
                reward += min(before_y - after_y, 3.0) * 0.18
            if after_y <= target_y + 8:
                reward += 0.08
            if result.name in {"mine_nearest_diamond", "branch_mine", "descend_to_diamond_layer"}:
                reward += 0.12

        if "achievement" in self.goal or "advancement" in self.goal:
            before_inv = self.inventory_counts(before.get("inventory", {}))
            after_inv = self.inventory_counts(after.get("inventory", {}))
            reward += max(0, after_inv["wood"] - before_inv["wood"]) * 0.25
            reward += max(0, after_inv["tools"] - before_inv["tools"]) * 0.5
            reward += max(0, after_inv["armor"] - before_inv["armor"]) * 0.35
            if result.name in {"engage_hostile", "mine_nearest_resource", "harvest_trees", "sprint_wander"}:
                reward += 0.08

        return reward

    def _record_experience(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        result: SkillResult,
        elapsed: float,
        decision_source: str,
        decision_reason: str,
    ) -> None:
        payload = {
            "time": time.time(),
            "step": self.policy.steps,
            "state": state,
            "action": action,
            "decision_source": decision_source,
            "decision_reason": decision_reason,
            "reward": reward,
            "next_state": next_state,
            "success": result.success,
            "details": result.details,
            "elapsed": elapsed,
            "epsilon": self.policy.epsilon,
            "goal": self.goal,
        }
        self._experience_buffer.append(payload)
        if len(self._experience_buffer) >= int(AUTONOMOUS_PARAMS["experience_flush_every"]):
            self._flush_experience()

    def _flush_experience(self) -> None:
        if not self._experience_buffer:
            return
        os.makedirs(os.path.dirname(self.experience_log), exist_ok=True)
        with open(self.experience_log, "a", encoding="utf-8") as handle:
            for payload in self._experience_buffer:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._experience_buffer.clear()

    def _remember_position(self, observation: Dict[str, Any]) -> None:
        self.position_history.append(self.position(observation.get("player", {})))

    def is_stuck(self) -> bool:
        window = int(AUTONOMOUS_PARAMS["stuck_window"])
        if len(self.position_history) < window:
            return False
        recent = list(self.position_history)[-window:]
        traveled = sum(self.distance(a, b) for a, b in zip(recent, recent[1:]))
        displacement = self.distance(recent[0], recent[-1])
        return traveled < float(AUTONOMOUS_PARAMS["stuck_distance"]) and displacement < float(AUTONOMOUS_PARAMS["stuck_distance"])

    def too_many_recent_failures(self) -> bool:
        return len(self.recent_failures) == self.recent_failures.maxlen and all(self.recent_failures)

    @staticmethod
    def in_high_danger(observation: Dict[str, Any]) -> bool:
        danger = str(observation.get("threats", {}).get("danger_level", "none"))
        player = observation.get("player", {})
        return danger in {"critical", "high"} or player.get("is_in_lava") or player.get("is_on_fire")

    @staticmethod
    def inventory_counts(inventory: Dict[str, Any]) -> Dict[str, int]:
        total = 0
        wood = 0
        tools = 0
        armor = 0
        for item in inventory.get("items", []):
            name = str(item.get("item", ""))
            count = int(item.get("count", 0))
            total += count
            if any(term in name for term in WOOD_TERMS):
                wood += count
            if any(term in name for term in TOOL_TERMS):
                tools += count
            if any(term in name for term in ARMOR_TERMS):
                armor += count
        return {"total": total, "wood": wood, "tools": tools, "armor": armor}

    @staticmethod
    def inventory_item_count(inventory: Dict[str, Any], terms: Iterable[str]) -> int:
        normalized_terms = tuple(str(term).lower() for term in terms)
        total = 0
        for item in inventory.get("items", []):
            name = str(item.get("item", "")).lower()
            if any(term in name for term in normalized_terms):
                total += int(item.get("count", 0))
        return total

    @staticmethod
    def nearby_block_matches(observation: Dict[str, Any], terms: Iterable[str]) -> bool:
        normalized_terms = tuple(str(term).lower() for term in terms)
        for block in observation.get("nearby_blocks", []):
            name = str(block.get("block", "")).lower()
            if any(term in name for term in normalized_terms):
                return True
        return False

    @staticmethod
    def closest_hostile_nearby(observation: Dict[str, Any]) -> bool:
        return any(BasicMechanics.is_hostile(entity) for entity in observation.get("nearby_entities", []))

    @staticmethod
    def normalize_goal(goal: str) -> str:
        return " ".join(str(goal or "").strip().lower().replace("_", " ").split())

    @staticmethod
    def position(player: Dict[str, Any]) -> Tuple[float, float, float]:
        position = player.get("position", {})
        return (float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0)))

    @staticmethod
    def distance(a: Iterable[float], b: Iterable[float]) -> float:
        ax, ay, az = a
        bx, by, bz = b
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    @staticmethod
    def bucket(value: float, cuts: Tuple[float, ...], labels: Tuple[str, ...]) -> str:
        for cut, label in zip(cuts, labels):
            if value < cut:
                return label
        return labels[-1]
