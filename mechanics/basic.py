import asyncio
import math
import random
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from ..bridge import BridgeClient, BridgeError
    from ..combat import CombatTactics
    from ..navigation import TerrainAStar
    from ..perception.world_state import build_world_state
except ImportError:
    from bridge import BridgeClient, BridgeError
    from combat import CombatTactics
    from navigation import TerrainAStar
    from perception.world_state import build_world_state


HOSTILE_KEYWORDS = (
    "zombie", "skeleton", "creeper", "spider", "enderman", "witch", "slime",
    "drowned", "husk", "stray", "phantom", "pillager", "vindicator", "ravager",
    "warden", "blaze", "ghast", "guardian", "shulker", "silverfish", "magma_cube",
    "hoglin", "zoglin", "piglin_brute",
)

RESOURCE_TERMS = (
    "log", "ore", "coal", "iron", "copper", "gold", "diamond", "redstone",
    "lapis", "emerald", "ancient_debris",
)

DANGEROUS_BLOCK_TERMS = ("lava", "fire", "magma_block", "cactus", "powder_snow", "sweet_berry_bush")
FLUID_BLOCK_TERMS = ("water", "lava")
UNBREAKABLE_BLOCK_TERMS = (
    "bedrock",
    "barrier",
    "command_block",
    "end_portal",
    "end_gateway",
    "jigsaw",
    "structure_block",
)


@dataclass
class SkillResult:
    name: str
    success: bool = True
    reward_hint: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class BasicMechanics:
    """
    Player-like Minecraft primitives built on bridge commands.

    Skills are intentionally small and noisy so a learner can compose them like a
    new player: look, walk, jump, scan, target, mine, collect, fight, flee, and
    harvest trees through ordinary bridge movement and mining commands.
    """

    SKILLS = (
        "survival_check",
        "respawn",
        "eat_food",
        "select_weapon",
        "adaptive_combat",
        "engage_hostile",
        "kite_hostile",
        "recover_safely",
        "idle",
        "look_around",
        "scan_environment",
        "visual_scan",
        "query_progression",
        "advancement_status",
        "wander",
        "sprint_wander",
        "jump_forward",
        "scan_logs",
        "mine_nearest_log",
        "dig_toward_surface",
        "mine_nearest_stone",
        "dig_to_stone",
        "mine_nearest_diamond",
        "mine_nearest_resource",
        "descend_to_diamond_layer",
        "branch_mine",
        "collect_item",
        "collect_visible_item",
        "flee_hostile",
        "attack_target",
        "practice_place",
        "harvest_trees",
        "craft_planks",
        "craft_sticks",
        "craft_crafting_table",
        "open_crafting_table",
        "craft_wooden_pickaxe",
        "craft_stone_pickaxe",
        "craft_furnace",
        "craft_torches",
    )

    def __init__(self, client: BridgeClient, step_delay: float = 0.15) -> None:
        self.client = client
        self.step_delay = step_delay
        self.combat_tactics = CombatTactics()

    async def observe(self) -> Dict[str, Any]:
        return await self.client.request("get_full_state")

    async def run(self, skill: str, observation: Optional[Dict[str, Any]] = None) -> SkillResult:
        observation = observation or await self.observe()
        method = getattr(self, f"skill_{skill}", None)
        if method is None and (skill.startswith("craft:") or skill.startswith("craft/")):
            target = skill.split(":", 1)[1] if ":" in skill else skill.split("/", 1)[1]
            return await self.craft(target, skill_name=skill)
        if method is None:
            return SkillResult(skill, success=False, reward_hint=-0.2, details={"error": "unknown_skill"})
        try:
            return await method(observation)
        except BridgeError as exc:
            return SkillResult(skill, success=False, reward_hint=-0.5, details={"error": str(exc)})

    async def stop_all(self) -> None:
        for action, payload in (
            ("stop_moving", {}),
            ("sprint", {"state": False}),
            ("sneak", {"state": False}),
        ):
            with suppress(BridgeError):
                await self.client.request(action, **payload)

    async def skill_idle(self, observation: Dict[str, Any]) -> SkillResult:
        await self.stop_all()
        await asyncio.sleep(self.step_delay)
        return SkillResult("idle", reward_hint=0.02)

    async def skill_survival_check(self, observation: Dict[str, Any]) -> SkillResult:
        skill, reason = self.choose_tactical_skill(observation, allow_resource_tasks=False)
        if skill and skill != "survival_check":
            result = await self.run(skill, observation)
            result.details.setdefault("tactical_reason", reason)
            result.reward_hint += 0.05
            return result
        await self.stop_all()
        return SkillResult("survival_check", reward_hint=0.08, details={"reason": reason or "stable"})

    async def skill_respawn(self, observation: Dict[str, Any]) -> SkillResult:
        player = observation.get("player", {})
        was_dead = float(player.get("health", 20.0)) <= 0.0 or bool(player.get("is_dead"))
        response = await self.client.request("respawn")
        await asyncio.sleep(1.5)
        refreshed = await self.observe()
        revived = float(refreshed.get("player", {}).get("health", 0.0)) > 0.0
        return SkillResult(
            "respawn",
            success=revived or not was_dead,
            reward_hint=0.4 if revived else -0.2,
            details={"was_dead": was_dead, "response": response, "health": refreshed.get("player", {}).get("health")},
        )

    async def skill_eat_food(self, observation: Dict[str, Any]) -> SkillResult:
        response = await self.client.request("eat_food")
        used = bool(response.get("used", True))
        return SkillResult(
            "eat_food",
            success=used,
            reward_hint=0.35 if used else -0.03,
            details=response,
        )

    async def skill_select_weapon(self, observation: Dict[str, Any]) -> SkillResult:
        response = await self.client.request("select_best_tool", purpose="weapon")
        selected = bool(response.get("selected", False))
        return SkillResult(
            "select_weapon",
            success=selected,
            reward_hint=0.08 if selected else -0.02,
            details=response,
        )

    async def skill_adaptive_combat(self, observation: Dict[str, Any]) -> SkillResult:
        world = build_world_state(observation)
        decision = self.combat_tactics.decide(world)
        details: Dict[str, Any] = {
            "strategy": decision.strategy,
            "reason": decision.reason,
            "threat_score": decision.threat_score,
            "mob_count": decision.mob_count,
            "desired_distance": decision.desired_distance,
            "metadata": decision.metadata,
        }
        target = decision.target
        if decision.escape_path is not None:
            details["escape_path"] = {
                "reached": decision.escape_path.reached,
                "cost": decision.escape_path.cost,
                "explored": decision.escape_path.explored,
                "path": [
                    {"x": x, "y": y, "z": z}
                    for x, y, z in decision.escape_path.path[:8]
                ],
            }

        if decision.strategy == "none":
            scan = await self.skill_scan_environment(observation)
            scan.name = "adaptive_combat"
            scan.details.setdefault("combat_decision", details)
            return scan

        await self.client.request("select_best_tool", purpose="weapon")

        if decision.strategy == "flee":
            if decision.escape_path is not None and decision.escape_path.next_waypoint is not None:
                waypoint = decision.escape_path.next_waypoint
                nav = await self.follow_path(decision.escape_path.path, sprint=True, max_waypoints=4, timeout_per_waypoint=1.4)
                details["path_navigation"] = nav
                details["first_waypoint"] = {"x": waypoint[0], "y": waypoint[1], "z": waypoint[2]}
            else:
                details["flee"] = await self.client.request("flee", radius=18.0, duration=1.0)
                await asyncio.sleep(0.9)
            await self.stop_all()
            return SkillResult("adaptive_combat", reward_hint=0.35, details=details)

        if target is None:
            details["fallback"] = await self.client.request("flee", radius=12.0, duration=0.7)
            await asyncio.sleep(0.7)
            await self.stop_all()
            return SkillResult("adaptive_combat", reward_hint=0.16, details=details)

        await self.client.request("aim_at_entity", entity_id=int(target.entity_id))

        if decision.strategy == "kite":
            if target.distance < decision.desired_distance:
                await self.client.request("move", forward=-0.35, strafe=decision.strafe)
            else:
                await self.client.request("move", forward=0.15, strafe=decision.strafe)
            await asyncio.sleep(random.uniform(0.22, 0.48))
            attack = await self.client.request("auto_attack", radius=8.0, approach=False, select_weapon=True)
            await asyncio.sleep(decision.attack_interval)
            await self.stop_all()
            details["attack"] = attack
            return SkillResult("adaptive_combat", reward_hint=0.24 if attack.get("attacked") else 0.16, details=details)

        if decision.strategy == "track":
            nav = await self.approach_position(
                int(round(target.pos[0])),
                int(round(target.pos[1])),
                int(round(target.pos[2])),
                stop_distance=decision.desired_distance,
                timeout=4.0,
                sprint=world.food > 8,
            )
            details["navigation"] = nav
            return SkillResult("adaptive_combat", reward_hint=0.14, details=details)

        if target.distance > 4.25:
            nav = await self.approach_position(
                int(round(target.pos[0])),
                int(round(target.pos[1])),
                int(round(target.pos[2])),
                stop_distance=3.2,
                timeout=3.8,
                sprint=world.food > 8,
            )
            details["navigation"] = nav

        await self.client.request("aim_at_entity", entity_id=int(target.entity_id))
        await asyncio.sleep(decision.attack_interval)
        attack = await self.client.request("auto_attack", radius=8.0, approach=True, select_weapon=True)
        await asyncio.sleep(random.uniform(0.08, 0.16))
        await self.stop_all()
        details["attack"] = attack
        return SkillResult(
            "adaptive_combat",
            success=bool(attack.get("attacked", False) or attack.get("approaching", False)),
            reward_hint=0.38 if attack.get("attacked") else 0.18,
            details=details,
        )

    async def skill_engage_hostile(self, observation: Dict[str, Any]) -> SkillResult:
        await self.client.request("select_best_tool", purpose="weapon")
        response = await self.client.request("auto_attack", radius=10.0, approach=True, select_weapon=True)
        attacked = bool(response.get("attacked", False))
        approaching = bool(response.get("approaching", False))
        return SkillResult(
            "engage_hostile",
            success=attacked or approaching,
            reward_hint=0.35 if attacked else 0.12 if approaching else -0.04,
            details=response,
        )

    async def skill_kite_hostile(self, observation: Dict[str, Any]) -> SkillResult:
        hostile = self.closest_entity(observation, self.is_hostile)
        if hostile is None:
            return SkillResult("kite_hostile", success=False, reward_hint=-0.02)
        await self.client.request("aim_at_entity", entity_id=int(hostile["id"]))
        await self.client.request("move", forward=-0.25, strafe=random.choice((-0.75, 0.75)))
        await asyncio.sleep(random.uniform(0.25, 0.55))
        await self.client.request("auto_attack", radius=8.0, approach=False, select_weapon=True)
        await self.stop_all()
        return SkillResult("kite_hostile", reward_hint=0.18, details={"target": hostile.get("type")})

    async def skill_recover_safely(self, observation: Dict[str, Any]) -> SkillResult:
        await self.client.request("sneak", state=True)
        eat_result: Optional[Dict[str, Any]] = None
        if self.should_eat(observation):
            try:
                eat_result = await self.client.request("eat_food")
            except BridgeError:
                eat_result = None
        await asyncio.sleep(random.uniform(0.25, 0.55))
        await self.stop_all()
        return SkillResult(
            "recover_safely",
            reward_hint=0.14 if eat_result else 0.06,
            details={"ate": bool(eat_result and eat_result.get("used", True))},
        )

    async def skill_look_around(self, observation: Dict[str, Any]) -> SkillResult:
        await self.client.request(
            "turn",
            delta_yaw=random.uniform(-75.0, 75.0),
            delta_pitch=random.uniform(-12.0, 12.0),
        )
        await asyncio.sleep(self.step_delay)
        return SkillResult("look_around", reward_hint=0.05)

    async def skill_scan_environment(self, observation: Dict[str, Any]) -> SkillResult:
        vision = await self.client.request("get_vision", rays=11, distance=48.0, fov=90.0)
        threats = await self.client.request("get_threats", radius=18.0)
        danger = threats.get("threats", {}).get("danger_level", "none")
        return SkillResult(
            "scan_environment",
            reward_hint=0.12 if danger == "none" else 0.18,
            details={"vision": vision.get("vision", {}), "threats": threats.get("threats", {})},
        )

    async def skill_visual_scan(self, observation: Dict[str, Any]) -> SkillResult:
        summary = await self.client.request("get_visual_summary", rays=17, distance=72.0, fov=110.0)
        visual = summary.get("visual_summary", {})
        reward = 0.16
        if visual.get("resource_count", 0) > 0:
            reward += 0.12
        if visual.get("hazard_count", 0) > 0:
            reward += 0.08
        return SkillResult("visual_scan", reward_hint=reward, details=visual)

    async def skill_query_progression(self, observation: Dict[str, Any]) -> SkillResult:
        plan = await self.client.request("get_advancement_plan")
        recipes = await self.client.request("list_recipes")
        return SkillResult(
            "query_progression",
            reward_hint=0.12,
            details={"plan": plan.get("plan", {}), "recipes": recipes.get("supported_recipes", [])},
        )

    async def skill_advancement_status(self, observation: Dict[str, Any]) -> SkillResult:
        status = await self.client.request("get_advancements", limit=500)
        missing = int(status.get("missing_count", 0))
        completed = int(status.get("completed_count", 0))
        return SkillResult(
            "advancement_status",
            reward_hint=0.12 + min(completed, 20) * 0.002,
            details={"completed_count": completed, "missing_count": missing, "status": status},
        )

    async def skill_craft_planks(self, observation: Dict[str, Any]) -> SkillResult:
        count = self.inventory_item_count(observation.get("inventory", {}), ("planks",))
        if count >= 4:
            return SkillResult("craft_planks", reward_hint=0.18, details={"already_available": True, "count": count})
        return await self.craft_target("craft_planks", "planks", max_crafts=8, reward=0.35)

    async def skill_craft_sticks(self, observation: Dict[str, Any]) -> SkillResult:
        count = self.inventory_item_count(observation.get("inventory", {}), ("stick",))
        if count >= 2:
            return SkillResult("craft_sticks", reward_hint=0.14, details={"already_available": True, "count": count})
        if self.inventory_item_count(observation.get("inventory", {}), ("planks",)) < 2:
            await self.skill_craft_planks(observation)
        return await self.craft_target("craft_sticks", "stick", max_crafts=4, reward=0.28)

    async def skill_craft_crafting_table(self, observation: Dict[str, Any]) -> SkillResult:
        count = self.inventory_item_count(observation.get("inventory", {}), ("crafting_table",))
        if count >= 1:
            return SkillResult("craft_crafting_table", reward_hint=0.2, details={"already_available": True, "count": count})
        if self.inventory_item_count(observation.get("inventory", {}), ("planks",)) < 4:
            await self.skill_craft_planks(observation)
        return await self.craft_target("craft_crafting_table", "crafting_table", max_crafts=1, reward=0.45)

    async def skill_open_crafting_table(self, observation: Dict[str, Any]) -> SkillResult:
        table = await self.client.request("find_nearest_block", block_type="crafting_table", max_radius=8)
        if table.get("found") and float(table.get("distance", 99.0)) > 4.5:
            await self.client.request("navigate_to", x=int(table["x"]), y=int(table["y"]), z=int(table["z"]))
            await asyncio.sleep(0.8)
            await self.stop_all()

        try:
            response = await self.client.request("open_crafting_table", radius=8)
        except BridgeError as exc:
            response = {"status": "error", "error": str(exc), "reason": str(exc).split(":", 1)[0]}
        if response.get("opened"):
            await asyncio.sleep(0.25)
            container = await self.client.request("get_container")
            return SkillResult(
                "open_crafting_table",
                reward_hint=0.25,
                details={"open": response, "container": container.get("container", {})},
            )

        if response.get("reason") == "out_of_reach":
            return SkillResult("open_crafting_table", success=False, reward_hint=0.02, details=response)

        if self.inventory_item_count(observation.get("inventory", {}), ("crafting_table",)) <= 0:
            craft_result = await self.skill_craft_crafting_table(observation)
            if not craft_result.success:
                return craft_result

        placed = await self.place_inventory_item_nearby("crafting_table")
        if not placed:
            return SkillResult("open_crafting_table", success=False, reward_hint=-0.04, details=response)

        await asyncio.sleep(0.25)
        try:
            response = await self.client.request("open_crafting_table", radius=8)
        except BridgeError as exc:
            response = {"status": "error", "error": str(exc), "reason": str(exc).split(":", 1)[0]}
        if response.get("opened"):
            await asyncio.sleep(0.25)
        return SkillResult("open_crafting_table", success=bool(response.get("opened")), reward_hint=0.22, details=response)

    async def skill_craft_wooden_pickaxe(self, observation: Dict[str, Any]) -> SkillResult:
        count = self.inventory_item_count(observation.get("inventory", {}), ("wooden_pickaxe",))
        if count >= 1:
            return SkillResult("craft_wooden_pickaxe", reward_hint=0.35, details={"already_available": True, "count": count})
        if self.inventory_item_count(observation.get("inventory", {}), ("stick",)) < 2:
            await self.skill_craft_sticks(observation)
        if self.inventory_item_count(observation.get("inventory", {}), ("planks",)) < 3:
            await self.skill_craft_planks(observation)
        if self.inventory_item_count(observation.get("inventory", {}), ("crafting_table",)) <= 0:
            await self.skill_craft_crafting_table(observation)
        table = await self.skill_open_crafting_table(await self.observe())
        if not table.success:
            return SkillResult("craft_wooden_pickaxe", success=False, reward_hint=0.02, details={"crafting_table_step": table.details})
        return await self.craft_target("craft_wooden_pickaxe", "wooden_pickaxe", max_crafts=1, reward=0.75)

    async def skill_craft_stone_pickaxe(self, observation: Dict[str, Any]) -> SkillResult:
        count = self.inventory_item_count(observation.get("inventory", {}), ("stone_pickaxe",))
        if count >= 1:
            return SkillResult("craft_stone_pickaxe", reward_hint=0.45, details={"already_available": True, "count": count})
        if self.inventory_item_count(observation.get("inventory", {}), ("stick",)) < 2:
            await self.skill_craft_sticks(observation)
        if self.inventory_item_count(observation.get("inventory", {}), ("cobblestone", "cobbled_deepslate")) < 3:
            mine = await self.skill_mine_nearest_stone(observation, desired_count=3)
            mine.details.setdefault("crafting_need", "cobblestone")
            return mine
        table = await self.skill_open_crafting_table(await self.observe())
        if not table.success:
            return SkillResult("craft_stone_pickaxe", success=False, reward_hint=0.02, details={"crafting_table_step": table.details})
        return await self.craft_target("craft_stone_pickaxe", "stone_pickaxe", max_crafts=1, reward=1.0)

    async def skill_craft_furnace(self, observation: Dict[str, Any]) -> SkillResult:
        count = self.inventory_item_count(observation.get("inventory", {}), ("furnace",))
        if count >= 1:
            return SkillResult("craft_furnace", reward_hint=0.35, details={"already_available": True, "count": count})
        if self.inventory_item_count(observation.get("inventory", {}), ("cobblestone", "cobbled_deepslate")) < 8:
            mine = await self.skill_mine_nearest_stone(observation, desired_count=8)
            mine.details.setdefault("crafting_need", "cobblestone")
            return mine
        table = await self.skill_open_crafting_table(await self.observe())
        if not table.success:
            return SkillResult("craft_furnace", success=False, reward_hint=0.02, details={"crafting_table_step": table.details})
        return await self.craft_target("craft_furnace", "furnace", max_crafts=1, reward=0.8)

    async def skill_craft_torches(self, observation: Dict[str, Any]) -> SkillResult:
        if self.inventory_item_count(observation.get("inventory", {}), ("stick",)) < 1:
            await self.skill_craft_sticks(observation)
        return await self.craft_target("craft_torches", "torch", max_crafts=4, reward=0.35)

    async def skill_wander(self, observation: Dict[str, Any]) -> SkillResult:
        await self.client.request("turn", delta_yaw=random.uniform(-35.0, 35.0), delta_pitch=random.uniform(-4.0, 4.0))
        await self.client.request("move", forward=random.uniform(0.45, 1.0), strafe=random.uniform(-0.35, 0.35))
        await asyncio.sleep(random.uniform(0.25, 0.7))
        await self.stop_all()
        return SkillResult("wander", reward_hint=0.06)

    async def skill_sprint_wander(self, observation: Dict[str, Any]) -> SkillResult:
        target_yaw: Optional[float] = None
        target_pitch = 0.0
        visual_details: Dict[str, Any] = {}
        try:
            visual = await self.client.request("get_visual_summary", rays=17, distance=48.0, fov=120.0)
            summary = visual.get("visual_summary", {})
            openings = summary.get("openings", []) if isinstance(summary, dict) else []
            if openings:
                opening = max(openings, key=lambda ray: float(ray.get("hit_distance", 0.0)))
                target_yaw = float(opening.get("yaw", 0.0))
                target_pitch = max(-8.0, min(8.0, float(opening.get("pitch", 0.0))))
                visual_details = {
                    "recommended_focus": summary.get("recommended_focus"),
                    "opening_yaw": target_yaw,
                    "opening_distance": opening.get("hit_distance"),
                }
        except BridgeError:
            visual_details = {}

        await self.client.request("sprint", state=True)
        if target_yaw is None:
            await self.client.request("turn", delta_yaw=random.uniform(-25.0, 25.0), delta_pitch=random.uniform(-3.0, 3.0))
        else:
            await self.client.request("look", yaw=target_yaw, pitch=target_pitch)
        await self.client.request("move", forward=1.0, strafe=random.uniform(-0.2, 0.2))
        await asyncio.sleep(random.uniform(0.65, 1.25))
        await self.stop_all()
        return SkillResult("sprint_wander", reward_hint=0.06 if target_yaw is not None else 0.04, details=visual_details)

    async def skill_jump_forward(self, observation: Dict[str, Any]) -> SkillResult:
        await self.client.request("move", forward=0.9, strafe=0.0)
        await self.client.request("jump")
        await asyncio.sleep(0.35)
        await self.stop_all()
        return SkillResult("jump_forward", reward_hint=0.04)

    async def skill_scan_logs(self, observation: Dict[str, Any]) -> SkillResult:
        response = await self.client.request("find_nearest_block", block_type="log", max_radius=24)
        found = bool(response.get("found"))
        return SkillResult("scan_logs", success=found, reward_hint=0.1 if found else -0.03, details=response)

    async def skill_mine_nearest_log(self, observation: Dict[str, Any]) -> SkillResult:
        response = await self.client.request(
            "find_nearest_block",
            block_type="log",
            max_radius=32,
            reachable_only=True,
            exposed_only=True,
            avoid_undermining=True,
            max_vertical_difference=6,
            max_reach_candidate_distance=6.0,
        )
        if not response.get("found"):
            beacon = await self.client.request(
                "find_nearest_block",
                block_type="log",
                max_radius=64,
                avoid_undermining=True,
                max_vertical_difference=16,
            )
            if not beacon.get("found"):
                return SkillResult("mine_nearest_log", success=False, reward_hint=-0.08, details=response)
            navigation = await self.approach_position(
                int(beacon["x"]),
                int(beacon["y"]),
                int(beacon["z"]),
                stop_distance=4.0,
                timeout=18.0,
                sprint=True,
            )
            retry = await self.client.request(
                "find_nearest_block",
                block_type="log",
                max_radius=24,
                reachable_only=True,
                exposed_only=True,
                avoid_undermining=True,
                max_vertical_difference=6,
                max_reach_candidate_distance=6.0,
            )
            if not retry.get("found"):
                distance_delta = float(navigation.get("start_distance", 0.0)) - float(navigation.get("final_distance", 0.0))
                escape: Optional[SkillResult] = None
                if distance_delta < 2.0:
                    escape = await self.dig_toward_target(await self.observe(), beacon)
                    retry = await self.client.request(
                        "find_nearest_block",
                        block_type="log",
                        max_radius=24,
                        reachable_only=True,
                        exposed_only=True,
                        avoid_undermining=True,
                        max_vertical_difference=6,
                        max_reach_candidate_distance=6.0,
                    )
                    if retry.get("found"):
                        observation = await self.observe()
                        response = retry
                    else:
                        navigation["escape"] = {
                            "success": escape.success,
                            "reward_hint": escape.reward_hint,
                            "details": escape.details,
                        }
                if not retry.get("found"):
                    return SkillResult(
                        "mine_nearest_log",
                        success=False,
                        reward_hint=0.08 if distance_delta >= 2.0 or bool(escape and escape.success) else -0.04,
                        details={"phase": "approach_distant_log", "initial_scan": response, "beacon": beacon, "navigation": navigation, "retry": retry},
                    )
            if retry.get("found"):
                observation = await self.observe()
                response = retry

        return await self.mine_known_block(
            "mine_nearest_log",
            observation,
            response,
            mine_attempts=30,
            mine_delay=0.15,
            approach_reward=0.12,
            mine_reward=0.45,
            stop_distance=3.6,
            timeout=10.0,
        )

    async def skill_dig_toward_surface(self, observation: Dict[str, Any]) -> SkillResult:
        beacon: Dict[str, Any] = {}
        try:
            beacon = await self.client.request(
                "find_nearest_block",
                block_type="log",
                max_radius=64,
                avoid_undermining=True,
                max_vertical_difference=24,
            )
        except BridgeError:
            beacon = {}
        return await self.dig_toward_target(observation, beacon if beacon.get("found") else None)

    async def skill_mine_nearest_resource(self, observation: Dict[str, Any]) -> SkillResult:
        for term in ("diamond_ore", "iron_ore", "coal_ore", "copper_ore", "log"):
            response = await self.client.request(
                "find_nearest_block",
                block_type=term,
                max_radius=32,
                reachable_only=True,
                exposed_only=True,
                avoid_undermining=True,
                max_vertical_difference=4,
                max_reach_candidate_distance=6.25,
            )
            if response.get("found"):
                return await self.mine_known_block(
                    "mine_nearest_resource",
                    observation,
                    response,
                    mine_attempts=24,
                    mine_delay=0.14,
                    approach_reward=0.16,
                    mine_reward=0.5,
                    stop_distance=3.4,
                    timeout=10.0,
                )
        return SkillResult("mine_nearest_resource", success=False, reward_hint=-0.05)

    async def skill_mine_nearest_stone(self, observation: Dict[str, Any], desired_count: int = 3) -> SkillResult:
        inventory = observation.get("inventory", {})
        existing_stone = self.inventory_item_count(inventory, ("cobblestone", "cobbled_deepslate"))
        if existing_stone >= desired_count:
            return SkillResult(
                "mine_nearest_stone",
                reward_hint=0.18,
                details={"already_available": True, "count": existing_stone, "threshold": desired_count},
            )
        for term in ("stone", "deepslate"):
            response = await self.client.request(
                "find_nearest_block",
                block_type=term,
                max_radius=24,
                reachable_only=True,
                exposed_only=True,
                avoid_undermining=True,
                max_vertical_difference=3,
                max_reach_candidate_distance=6.25,
            )
            if not response.get("found"):
                continue
            return await self.mine_known_block(
                "mine_nearest_stone",
                observation,
                response,
                mine_attempts=26,
                mine_delay=0.13,
                approach_reward=0.14,
                mine_reward=0.45,
                stop_distance=3.0,
                timeout=10.0,
            )

        buried_attempts: List[Dict[str, Any]] = []
        for term in ("stone", "deepslate"):
            response = await self.client.request(
                "find_nearest_block",
                block_type=term,
                max_radius=24,
                reachable_only=False,
                exposed_only=False,
                avoid_undermining=True,
                max_vertical_difference=8,
            )
            if not response.get("found"):
                continue
            result = await self.expose_and_mine_buried_stone(observation, response)
            if result.success:
                return result
            buried_attempts.append(result.details)

        dig = await self.skill_dig_to_stone(observation)
        dig.name = "mine_nearest_stone"
        if buried_attempts:
            dig.details["buried_stone_attempts"] = buried_attempts
        return dig

    async def expose_and_mine_buried_stone(self, observation: Dict[str, Any], block_response: Dict[str, Any]) -> SkillResult:
        block = (int(block_response["x"]), int(block_response["y"]), int(block_response["z"]))
        details: Dict[str, Any] = {"phase": "expose_buried_stone", **block_response, "access_steps": []}
        before_inventory = await self.safe_inventory()
        before_count = self.inventory_item_count(before_inventory, ("cobblestone", "cobbled_deepslate"))

        await self.client.request("close_container")
        await self.client.request("sneak", state=True)
        try:
            current = self.player_position(await self.observe())
            horizontal_distance = math.hypot((block[0] + 0.5) - current[0], (block[2] + 0.5) - current[2])
            if horizontal_distance > 3.25:
                navigation = await self.approach_position(
                    block[0],
                    int(math.floor(current[1])),
                    block[2],
                    stop_distance=2.4,
                    timeout=6.0,
                    sprint=horizontal_distance > 8.0,
                )
                details["approach"] = navigation
                current = self.player_position(await self.observe())

            top_y = max(int(math.floor(current[1])) + 1, block[1] + 1)
            if top_y - block[1] > 8:
                return SkillResult("mine_nearest_stone", success=False, reward_hint=-0.03, details=details)

            for access_y in range(top_y, block[1], -1):
                refreshed = await self.observe()
                step = await self.mine_block_until_changed(
                    refreshed,
                    (block[0], access_y, block[2]),
                    attempts=38,
                    delay=0.09,
                )
                details["access_steps"].append(step)
                probe = f"{step.get('from', '')} {step.get('to', '')}".lower()
                if any(term in probe for term in FLUID_BLOCK_TERMS):
                    return SkillResult("mine_nearest_stone", success=False, reward_hint=-0.05, details=details)

            refreshed = await self.observe()
            target_info = await self.client.request("get_block_info", x=block[0], y=block[1], z=block[2])
            target_block = str(target_info.get("block") or block_response.get("block") or "minecraft:stone")
            mine_response = dict(block_response)
            mine_response["block"] = target_block
            mine_response["distance"] = self.distance(self.player_position(refreshed), (block[0] + 0.5, block[1] + 0.5, block[2] + 0.5))
            result = await self.mine_known_block(
                "mine_nearest_stone",
                refreshed,
                mine_response,
                mine_attempts=36,
                mine_delay=0.12,
                approach_reward=0.12,
                mine_reward=0.55,
                stop_distance=3.0,
                timeout=6.0,
                reach_distance=5.8,
            )
            details["mine"] = result.details
            after_inventory = await self.safe_inventory()
            after_count = self.inventory_item_count(after_inventory, ("cobblestone", "cobbled_deepslate"))
            details["inventory_before_count"] = before_count
            details["inventory_after_count"] = after_count
            details["inventory_delta"] = after_count - before_count
            return SkillResult(
                "mine_nearest_stone",
                success=result.success or after_count > before_count,
                reward_hint=0.55 if result.success or after_count > before_count else result.reward_hint,
                details=details,
            )
        finally:
            with suppress(BridgeError):
                await self.client.request("sneak", state=False)

    async def skill_dig_to_stone(self, observation: Dict[str, Any]) -> SkillResult:
        dug: List[Dict[str, Any]] = []
        await self.client.request("close_container")
        await self.client.request("sneak", state=True)
        stuck_steps = 0
        timed_out = False
        attempted_fronts: set[Tuple[int, int, int]] = set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 22.0
        try:
            for _ in range(10):
                if loop.time() >= deadline:
                    timed_out = True
                    break
                observation = await self.observe()
                for term in ("stone", "deepslate"):
                    response = await self.client.request(
                        "find_nearest_block",
                        block_type=term,
                        max_radius=8,
                        reachable_only=True,
                        exposed_only=True,
                        avoid_undermining=True,
                        max_vertical_difference=3,
                        max_reach_candidate_distance=6.25,
                    )
                    if response.get("found"):
                        result = await self.mine_known_block(
                            "dig_to_stone",
                            observation,
                            response,
                            mine_attempts=24,
                            mine_delay=0.13,
                            approach_reward=0.12,
                            mine_reward=0.55,
                            stop_distance=3.0,
                            timeout=6.0,
                        )
                        result.details["dug_stair_steps"] = dug
                        return result

                player = observation.get("player", {})
                x, y, z = self.player_position(observation)
                yaw = float(player.get("yaw", 0.0))
                if stuck_steps >= 2:
                    yaw += random.choice((-90.0, 90.0, 135.0))
                    await self.client.request("look", yaw=yaw, pitch=float(player.get("pitch", 0.0)))
                step_x, step_z = self.forward_step(yaw)
                base_x, base_y, base_z = int(round(x)), int(math.floor(y)), int(round(z))
                lower_front = (base_x + step_x, base_y - 1, base_z + step_z)
                if lower_front in attempted_fronts:
                    stuck_steps += 1
                    await self.client.request("turn", delta_yaw=random.choice((-110.0, 110.0)), delta_pitch=0.0)
                    await asyncio.sleep(0.15)
                    continue
                attempted_fronts.add(lower_front)
                stair_blocks = (
                    (base_x + step_x, base_y + 1, base_z + step_z),
                    (base_x + step_x, base_y, base_z + step_z),
                    lower_front,
                )

                mined_any = False
                blocked_by_fluid = False
                for block in stair_blocks:
                    if loop.time() >= deadline:
                        timed_out = True
                        break
                    mined = await self.mine_block_until_changed(observation, block, attempts=22, delay=0.1)
                    dug.append(mined)
                    mined_any = mined_any or bool(mined.get("cleared") or mined.get("changed"))
                    fluid_probe = f"{mined.get('from', '')} {mined.get('to', '')}".lower()
                    blocked_by_fluid = blocked_by_fluid or any(term in fluid_probe for term in FLUID_BLOCK_TERMS)
                if timed_out:
                    break
                if blocked_by_fluid:
                    stuck_steps += 2
                    await self.client.request("turn", delta_yaw=random.choice((-120.0, 120.0)), delta_pitch=0.0)
                    await self.client.request(
                        "move",
                        forward=-0.35,
                        strafe=random.choice((-0.45, 0.45)),
                        duration=0.45,
                    )
                    await asyncio.sleep(0.25)
                    await self.stop_all()
                    continue

                before_move = self.player_position(await self.observe())
                await self.client.request(
                    "navigate_to",
                    x=lower_front[0],
                    y=lower_front[1],
                    z=lower_front[2],
                    stop_distance=0.55,
                    timeout=2.4,
                    sprint=False,
                )
                await asyncio.sleep(0.85 if mined_any else 0.35)
                await self.client.request("move", forward=0.55, strafe=0.0, duration=0.45)
                await asyncio.sleep(0.5)
                await self.stop_all()
                after_move = self.player_position(await self.observe())
                if self.distance(before_move, after_move) < 0.25:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
        finally:
            with suppress(BridgeError):
                await self.client.request("sneak", state=False)

        return SkillResult(
            "dig_to_stone",
            success=False,
            reward_hint=-0.04,
            details={"dug_stair_steps": dug, "timed_out": timed_out},
        )

    async def skill_mine_nearest_diamond(self, observation: Dict[str, Any]) -> SkillResult:
        response = await self.client.request(
            "find_nearest_block",
            block_type="diamond_ore",
            max_radius=48,
            reachable_only=True,
            exposed_only=True,
            avoid_undermining=True,
            max_vertical_difference=8,
            max_reach_candidate_distance=6.25,
        )
        if not response.get("found"):
            y = self.player_position(observation)[1]
            if y > -45:
                result = await self.skill_descend_to_diamond_layer(observation)
                result.details.setdefault("diamond_scan", "not_found")
                return result
            result = await self.skill_branch_mine(observation)
            result.details.setdefault("diamond_scan", "not_found")
            return result

        block = (int(response["x"]), int(response["y"]), int(response["z"]))
        await self.client.request("select_best_tool", x=block[0], y=block[1], z=block[2])
        tool_check = await self.client.request("check_tool_validity", x=block[0], y=block[1], z=block[2])
        if tool_check.get("requires_tool") and not tool_check.get("can_harvest"):
            return SkillResult(
                "mine_nearest_diamond",
                success=False,
                reward_hint=0.15,
                details={"phase": "need_iron_pickaxe", "diamond": response, "tool_check": tool_check},
            )

        return await self.mine_known_block(
            "mine_nearest_diamond",
            observation,
            response,
            mine_attempts=32,
            mine_delay=0.15,
            approach_reward=0.6,
            mine_reward=2.0,
            stop_distance=3.2,
            timeout=12.0,
        )

    async def skill_descend_to_diamond_layer(self, observation: Dict[str, Any]) -> SkillResult:
        dug: List[Dict[str, Any]] = []
        start_y = self.player_position(observation)[1]
        await self.client.request("close_container")
        await self.client.request("sneak", state=True)
        stuck_steps = 0
        try:
            for _ in range(8):
                observation = await self.observe()
                x, y, z = self.player_position(observation)
                if y <= -54.0:
                    return SkillResult(
                        "descend_to_diamond_layer",
                        reward_hint=0.35,
                        details={"target_y": -54, "current_y": y, "dug": dug},
                    )

                player = observation.get("player", {})
                yaw = float(player.get("yaw", 0.0))
                if stuck_steps >= 2:
                    yaw += 35.0
                    await self.client.request("look", yaw=yaw, pitch=float(player.get("pitch", 0.0)))
                step_x, step_z = self.forward_step(yaw)
                base_x, base_y, base_z = int(round(x)), int(math.floor(y)), int(round(z))
                front = (base_x + step_x, base_y, base_z + step_z)
                head = (base_x + step_x, base_y + 1, base_z + step_z)
                lower_front = (base_x + step_x, base_y - 1, base_z + step_z)

                for block in (head, front, lower_front):
                    dug.append(await self.mine_block_until_changed(observation, block, attempts=48, delay=0.1))

                before_move = self.player_position(await self.observe())
                await self.client.request(
                    "navigate_to",
                    x=lower_front[0],
                    y=lower_front[1],
                    z=lower_front[2],
                    stop_distance=0.55,
                    timeout=3.5,
                    sprint=False,
                )
                await asyncio.sleep(0.9)
                await self.client.request("move", forward=0.55, strafe=0.0, duration=0.45)
                await asyncio.sleep(0.5)
                await self.stop_all()
                after_move = self.player_position(await self.observe())
                if self.distance(before_move, after_move) < 0.25:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
        finally:
            with suppress(BridgeError):
                await self.client.request("sneak", state=False)

        current_y = self.player_position(await self.observe())[1]
        return SkillResult(
            "descend_to_diamond_layer",
            success=current_y < start_y,
            reward_hint=0.22,
            details={"target_y": -54, "current_y": current_y, "dug": dug},
        )

    async def skill_branch_mine(self, observation: Dict[str, Any]) -> SkillResult:
        dug: List[Dict[str, Any]] = []
        await self.client.request("close_container")
        for step in range(4):
            observation = await self.observe()
            diamond = await self.client.request(
                "find_nearest_block",
                block_type="diamond_ore",
                max_radius=12,
                reachable_only=True,
                exposed_only=True,
                avoid_undermining=True,
                max_vertical_difference=4,
                max_reach_candidate_distance=6.25,
            )
            if diamond.get("found"):
                result = await self.skill_mine_nearest_diamond(observation)
                result.details.setdefault("branch_steps", dug)
                return result

            player = observation.get("player", {})
            x, y, z = self.player_position(observation)
            yaw = float(player.get("yaw", 0.0))
            if step == 2:
                yaw += 90.0
                await self.client.request("look", yaw=yaw, pitch=float(player.get("pitch", 0.0)))
            step_x, step_z = self.forward_step(yaw)
            base_x, base_y, base_z = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
            columns = [(base_x + step_x, base_z + step_z)]
            if step_x and step_z:
                columns.extend(((base_x + step_x, base_z), (base_x, base_z + step_z)))
            targets = [
                (column_x, block_y, column_z)
                for column_x, column_z in columns
                for block_y in (base_y, base_y + 1)
            ]
            for block in targets:
                dug.append(await self.mine_block_until_changed(observation, block, attempts=54, delay=0.1))

            await self.client.request(
                "navigate_to",
                x=base_x + step_x,
                y=base_y,
                z=base_z + step_z,
                stop_distance=0.6,
                timeout=3.0,
                sprint=False,
            )
            await asyncio.sleep(0.8)
            await self.client.request("move", forward=0.6, strafe=0.0, duration=0.35)
            await asyncio.sleep(0.4)
            await self.stop_all()

        return SkillResult("branch_mine", reward_hint=0.28, details={"dug": dug})

    async def mine_known_block(
        self,
        skill_name: str,
        observation: Dict[str, Any],
        block_response: Dict[str, Any],
        *,
        mine_attempts: int,
        mine_delay: float,
        approach_reward: float,
        mine_reward: float,
        stop_distance: float,
        timeout: float,
        reach_distance: float = 5.0,
    ) -> SkillResult:
        block = (int(block_response["x"]), int(block_response["y"]), int(block_response["z"]))
        target_center = (block[0] + 0.5, block[1] + 0.5, block[2] + 0.5)
        player = self.player_position(observation)
        eye = (player[0], player[1] + 1.62, player[2])
        distance = float(block_response.get("distance", self.distance(player, target_center)))
        reach_to_target = float(block_response.get("reach_distance", self.distance(eye, target_center)))
        details: Dict[str, Any] = {"phase": "mine", **block_response, "initial_reach_distance": reach_to_target}

        if reach_to_target > reach_distance:
            navigation = await self.approach_position(
                block[0],
                block[1],
                block[2],
                stop_distance=stop_distance,
                timeout=timeout,
                sprint=distance > 8.0,
            )
            await asyncio.sleep(min(timeout, max(1.0, distance / 4.5)))
            await self.stop_all()

            observation = await self.observe()
            player = self.player_position(observation)
            distance = self.distance(player, target_center)
            eye = (player[0], player[1] + 1.62, player[2])
            reach_to_target = self.distance(eye, target_center)
            details.update(
                {
                    "phase": "approach",
                    "navigation": navigation,
                    "post_distance": distance,
                    "post_reach_distance": reach_to_target,
                }
            )
            if reach_to_target > reach_distance + 0.75:
                return SkillResult(skill_name, success=False, reward_hint=approach_reward, details=details)
            details["phase"] = "approach_then_mine"

        await self.look_at_block(observation, block)
        tool = await self.client.request("select_best_tool", x=block[0], y=block[1], z=block[2])
        details["tool"] = tool

        final_block: Dict[str, Any] = {}
        progress: Dict[str, Any] = {}
        mine_error: Optional[str] = None
        for _ in range(max(1, mine_attempts)):
            try:
                await self.client.request("mine", x=block[0], y=block[1], z=block[2])
            except BridgeError as exc:
                mine_error = str(exc)
                break
            await asyncio.sleep(mine_delay)
            final_block = await self.client.request("get_block_info", x=block[0], y=block[1], z=block[2])
            if final_block.get("is_air") or str(final_block.get("block", "")) != str(block_response.get("block", "")):
                break
            progress = await self.client.request("get_mining_progress", x=block[0], y=block[1], z=block[2])
            if progress.get("completed"):
                break

        if final_block:
            details["final_block"] = final_block
        if progress:
            details["progress"] = progress
        if mine_error:
            details["mine_error"] = mine_error
        mined = bool(final_block.get("is_air")) or (
            final_block and str(final_block.get("block", "")) != str(block_response.get("block", ""))
        ) or bool(progress.get("completed"))
        if mined and not mine_error:
            details["drop_collection"] = await self.collect_drops_near_block(block)
        return SkillResult(
            skill_name,
            success=mined and not mine_error,
            reward_hint=mine_reward if mined else mine_reward * 0.6 if not mine_error else approach_reward,
            details=details,
        )

    async def collect_drops_near_block(self, block: Tuple[int, int, int]) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        try:
            attempts.append(
                await self.client.request(
                    "navigate_to",
                    x=block[0],
                    y=block[1],
                    z=block[2],
                    stop_distance=0.55,
                    timeout=3.5,
                    sprint=False,
                )
            )
            await asyncio.sleep(0.75)

            for _ in range(3):
                observation = await self.observe()
                item = self.closest_entity(observation, lambda entity: entity.get("type") == "minecraft:item")
                if item is None or float(item.get("distance", 99.0)) > 5.0:
                    break
                attempts.append(
                    await self.client.request(
                        "navigate_to",
                        x=int(round(float(item["x"]))),
                        y=int(round(float(item["y"]))),
                        z=int(round(float(item["z"]))),
                        stop_distance=0.45,
                        timeout=3.0,
                        sprint=False,
                    )
                )
                await asyncio.sleep(0.55)
        except BridgeError as exc:
            attempts.append({"status": "error", "error": str(exc)})
        finally:
            await self.stop_all()
        return {"block": {"x": block[0], "y": block[1], "z": block[2]}, "attempts": attempts}

    async def skill_collect_item(
        self,
        observation: Dict[str, Any],
        item_terms: Optional[Iterable[str]] = None,
    ) -> SkillResult:
        normalized_terms = tuple(str(term).replace("minecraft:", "").lower() for term in item_terms or ())

        def matches_item(entity: Dict[str, Any]) -> bool:
            if entity.get("type") != "minecraft:item":
                return False
            if not normalized_terms:
                return True
            item_name = str(entity.get("item_name", "")).replace("minecraft:", "").lower()
            return any(term in item_name for term in normalized_terms)

        item = self.closest_entity(observation, matches_item)
        if item is None:
            return SkillResult("collect_item", success=False, reward_hint=-0.02)

        before_inventory = await self.safe_inventory()
        before_total = sum(int(stack.get("count", 0)) for stack in before_inventory.get("items", []))
        target_id = item.get("id")
        target_item_name = str(item.get("item_name", ""))
        target_terms = (target_item_name.replace("minecraft:", ""),) if target_item_name else ()
        before_target = self.inventory_item_count(before_inventory, target_terms) if target_terms else before_total

        for _ in range(5):
            await self.client.request(
                "navigate_to",
                x=int(round(float(item["x"]))),
                y=int(round(float(item["y"]))),
                z=int(round(float(item["z"]))),
                stop_distance=0.5,
                timeout=4.0,
                sprint=False,
            )
            await asyncio.sleep(0.9)

            refreshed = await self.observe()
            after_inventory = refreshed.get("inventory", {})
            after_total = sum(int(stack.get("count", 0)) for stack in after_inventory.get("items", []))
            after_target = self.inventory_item_count(after_inventory, target_terms) if target_terms else after_total
            item_still_present = any(
                entity.get("type") == "minecraft:item" and entity.get("id") == target_id
                for entity in refreshed.get("nearby_entities", [])
            )
            if after_target > before_target or after_total > before_total:
                await self.stop_all()
                return SkillResult(
                    "collect_item",
                    reward_hint=0.24,
                    details={
                        "entity_id": target_id,
                        "item_name": target_item_name,
                        "inventory_before": before_total,
                        "inventory_after": after_total,
                        "target_before": before_target,
                        "target_after": after_target,
                        "picked_up": True,
                    },
                )

            item = self.closest_entity(refreshed, matches_item)
            if item is None:
                break

        await self.stop_all()
        return SkillResult(
            "collect_item",
            success=False,
            reward_hint=-0.03,
            details={
                "entity_id": target_id,
                "item_name": target_item_name,
                "inventory_before": before_total,
                "target_before": before_target,
            },
        )

    async def skill_collect_visible_item(self, observation: Dict[str, Any]) -> SkillResult:
        result = await self.skill_collect_item(observation)
        result.details.setdefault("delegated_skill", result.name)
        result.name = "collect_visible_item"
        return result

    async def skill_flee_hostile(self, observation: Dict[str, Any]) -> SkillResult:
        hostile = self.closest_entity(observation, self.is_hostile)
        if hostile is None and not self.in_environmental_danger(observation):
            return SkillResult("flee_hostile", success=False, reward_hint=-0.01)
        response = await self.client.request("flee", radius=18.0, duration=0.9)
        await asyncio.sleep(0.8)
        await self.stop_all()
        return SkillResult(
            "flee_hostile",
            reward_hint=0.25,
            details={"from": hostile.get("type") if hostile else "hazard", "response": response},
        )

    async def skill_attack_target(self, observation: Dict[str, Any]) -> SkillResult:
        response = await self.client.request("auto_attack", radius=8.0, approach=True, select_weapon=True)
        await asyncio.sleep(0.25)
        return SkillResult(
            "attack_target",
            success=bool(response.get("attacked", False) or response.get("approaching", False)),
            reward_hint=0.3 if response.get("attacked") else 0.1,
            details=response,
        )

    async def skill_practice_place(self, observation: Dict[str, Any]) -> SkillResult:
        ray = await self.client.request("raycast", distance=4.5)
        if not ray.get("hit") or ray.get("hit_type") != "block":
            return SkillResult("practice_place", success=False, reward_hint=-0.03, details=ray)
        response = await self.client.request(
            "place",
            x=int(ray["x"]),
            y=int(ray["y"]),
            z=int(ray["z"]),
            facing=str(ray.get("face", "UP")),
        )
        return SkillResult("practice_place", reward_hint=0.05, details=response)

    async def skill_harvest_trees(self, observation: Dict[str, Any]) -> SkillResult:
        attempts: List[Dict[str, Any]] = []
        total_reward = 0.0
        for _ in range(8):
            response = await self.client.request("harvest", resource="wood", radius=24)
            attempts.append(response)
            phase = str(response.get("phase", ""))
            if not response.get("found", True):
                break
            if phase == "approach":
                await asyncio.sleep(1.0)
                await self.stop_all()
                total_reward += 0.12
                observation = await self.observe()
                continue
            if phase == "mine":
                await asyncio.sleep(0.45)
                total_reward += 0.18
                continue
            break

        success = any(str(attempt.get("phase", "")) in {"approach", "mine"} for attempt in attempts)
        return SkillResult(
            "harvest_trees",
            success=success,
            reward_hint=0.08 + total_reward if success else -0.06,
            details={"attempts": attempts},
        )

    async def craft(
        self,
        target: str,
        max_crafts: int = 1,
        *,
        skill_name: Optional[str] = None,
        reward: float = 0.35,
        verify: bool = True,
    ) -> SkillResult:
        skill = skill_name or f"craft_{target}"
        normalized_target = str(target).replace("minecraft:", "")
        before_inventory = await self.safe_inventory()
        before_count = self.inventory_item_count(before_inventory, (normalized_target,))
        query = await self.client.request("query_recipe", item=target)
        recipe = query.get("recipe", {}) if isinstance(query.get("recipe"), dict) else {}
        needs_table = bool(recipe.get("requires_crafting_table"))
        context = query.get("crafting_context", {}) if isinstance(query.get("crafting_context"), dict) else {}

        table_result: Optional[SkillResult] = None
        if needs_table and not context.get("using_crafting_table"):
            table_result = await self.skill_open_crafting_table(await self.observe())

        response = await self.client.request("craft", item=target, max_crafts=max_crafts)
        reason = str(response.get("reason", ""))
        if reason in {"crafting_table_required", "no_crafting_grid"}:
            table_result = await self.skill_open_crafting_table(await self.observe())
            if table_result.success:
                response = await self.client.request("craft", item=target, max_crafts=max_crafts)
                reason = str(response.get("reason", ""))

        crafted = bool(response.get("crafted", False))
        output_item = str(response.get("item") or recipe.get("result") or target)
        output_term = output_item.replace("minecraft:", "")
        output_before_count = self.inventory_item_count(before_inventory, (output_term,))
        expected_delta = int(response.get("output_count", 1)) if crafted else 0
        verified = False
        after_inventory: Dict[str, Any] = {}
        if crafted and verify:
            for delay in (0.2, 0.45, 0.9):
                await asyncio.sleep(delay)
                after_inventory = await self.safe_inventory()
                after_count = self.inventory_item_count(after_inventory, (output_term,))
                if after_count >= output_before_count + max(1, expected_delta):
                    verified = True
                    break
        reward_hint = reward if crafted else -0.05
        if reason == "crafting_table_required":
            reward_hint = 0.02
        details = {
            **response,
            "query": query,
            "verified_inventory_delta": verified,
            "inventory_before_count": before_count,
            "output_before_count": output_before_count,
            "inventory_after": after_inventory,
        }
        if table_result is not None:
            details["crafting_table_step"] = table_result.details
        return SkillResult(
            skill,
            success=crafted,
            reward_hint=reward_hint,
            details=details,
        )

    async def craft_target(self, skill_name: str, item: str, max_crafts: int, reward: float) -> SkillResult:
        return await self.craft(item, max_crafts=max_crafts, skill_name=skill_name, reward=reward)

    async def safe_inventory(self) -> Dict[str, Any]:
        try:
            response = await self.client.request("get_inventory")
            inventory = response.get("inventory", {})
            return inventory if isinstance(inventory, dict) else {}
        except (BridgeError, OSError, asyncio.TimeoutError):
            return {"items": []}

    async def place_inventory_item_nearby(self, term: str) -> bool:
        slot = await self.select_inventory_item(term)
        if slot < 0:
            return False
        ray = await self.client.request("raycast", distance=4.5)
        if not ray.get("hit") or ray.get("hit_type") != "block":
            await self.client.request("look", pitch=35.0, yaw=random.uniform(-180.0, 180.0))
            await asyncio.sleep(0.05)
            ray = await self.client.request("raycast", distance=4.5)
        if ray.get("hit") and ray.get("hit_type") == "block":
            await self.client.request(
                "place",
                x=int(ray["x"]),
                y=int(ray["y"]),
                z=int(ray["z"]),
                facing=str(ray.get("face", "UP")),
            )
            await asyncio.sleep(0.2)
            found = await self.client.request("find_nearest_block", block_type=term, max_radius=6)
            if found.get("found"):
                return True

        observation = await self.observe()
        px, py, pz = self.player_position(observation)
        base_x, base_y, base_z = int(math.floor(px)), int(math.floor(py)) - 1, int(math.floor(pz))
        candidates = (
            (base_x, base_y, base_z),
            (base_x + 1, base_y, base_z),
            (base_x - 1, base_y, base_z),
            (base_x, base_y, base_z + 1),
            (base_x, base_y, base_z - 1),
        )
        for x, y, z in candidates:
            try:
                await self.client.request("place", x=x, y=y, z=z, facing="UP")
            except BridgeError:
                continue
            await asyncio.sleep(0.2)
            found = await self.client.request("find_nearest_block", block_type=term, max_radius=6)
            if found.get("found"):
                return True
        return False

    async def select_inventory_item(self, term: str) -> int:
        found = await self.client.request("find_item", item=term)
        slots = found.get("found_slots", [])
        if not slots:
            return -1
        slot = int(slots[0].get("slot", -1))
        if 0 <= slot <= 8:
            await self.client.request("select_slot", slot=slot)
            return slot
        hotbar_slot = 0
        await self.client.request("swap_slots", slot1=slot, slot2=hotbar_slot)
        await self.client.request("select_slot", slot=hotbar_slot)
        return hotbar_slot

    def choose_tactical_skill(
        self,
        observation: Dict[str, Any],
        allow_resource_tasks: bool = True,
    ) -> Tuple[Optional[str], str]:
        player = observation.get("player", {})
        survival = observation.get("survival", {})
        combat = observation.get("combat", {})
        threats = observation.get("threats", {})

        if float(player.get("health", 20.0)) <= 0.0 or bool(player.get("is_dead")):
            return "respawn", "player_dead"
        if self.in_environmental_danger(observation):
            return "flee_hostile", "environmental_danger"
        if self.low_oxygen(player):
            return "jump_forward", "low_oxygen"
        if self.should_eat(observation):
            return "eat_food", "low_food_or_health"

        danger_level = str(threats.get("danger_level", survival.get("danger_level", "none")))
        has_hostile = self.closest_entity(observation, self.is_hostile) is not None
        if survival.get("should_flee") or danger_level in {"critical", "high"}:
            return "adaptive_combat" if has_hostile else "flee_hostile", f"danger_{danger_level}"
        if has_hostile and float(player.get("health", 20.0)) <= 10.0:
            return "adaptive_combat", "hurt_with_hostile"
        if combat.get("has_target") or has_hostile:
            return "adaptive_combat", "combat_target"

        if allow_resource_tasks:
            if self.closest_entity(observation, lambda e: e.get("type") == "minecraft:item") is not None:
                return "collect_visible_item", "nearby_item"
            if self.visible_resource_nearby(observation):
                return "mine_nearest_resource", "nearby_resource"

        return None, "no_override"

    async def follow_path(
        self,
        path: List[Tuple[int, int, int]],
        *,
        sprint: bool,
        max_waypoints: int = 8,
        timeout_per_waypoint: float = 1.6,
    ) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        for waypoint in path[1 : max(1, max_waypoints + 1)]:
            response = await self.client.request(
                "navigate_to",
                x=waypoint[0],
                y=waypoint[1],
                z=waypoint[2],
                stop_distance=0.8,
                timeout=timeout_per_waypoint,
                sprint=sprint,
            )
            attempts.append(response)
            await asyncio.sleep(min(0.9, timeout_per_waypoint))
            observation = await self.observe()
            if self.distance(self.player_position(observation), (waypoint[0] + 0.5, waypoint[1], waypoint[2] + 0.5)) <= 1.2:
                continue
        await self.stop_all()
        return {"attempts": attempts, "waypoints_used": len(attempts)}

    async def approach_position(
        self,
        x: int,
        y: int,
        z: int,
        *,
        stop_distance: float,
        timeout: float,
        sprint: bool,
    ) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        start_observation = await self.observe()
        start = self.player_position(start_observation)
        target = (x + 0.5, y + 0.5, z + 0.5)
        start_distance = self.distance(start, target)
        try:
            world = build_world_state(start_observation)
            plan = TerrainAStar(world).plan(
                world.player_block,
                (int(x), int(y), int(z)),
                max_nodes=4096,
                allow_partial=True,
            )
            if plan.path and len(plan.path) > 1:
                path_follow = await self.follow_path(
                    plan.path,
                    sprint=sprint and start_distance > 7.0 and world.food > 8.0,
                    max_waypoints=8,
                    timeout_per_waypoint=1.5,
                )
                after_path = self.player_position(await self.observe())
                path_distance = self.distance(after_path, target)
                attempts.append(
                    {
                        "phase": "terrain_astar",
                        "reached": path_distance <= stop_distance,
                        "plan_reached": plan.reached,
                        "plan_reason": plan.reason,
                        "plan_cost": plan.cost,
                        "explored": plan.explored,
                        "path_length": len(plan.path),
                        "final_distance": path_distance,
                        "path_follow": path_follow,
                    }
                )
                if path_distance <= stop_distance:
                    return {
                        "status": "success",
                        "reached": True,
                        "target_x": x,
                        "target_y": y,
                        "target_z": z,
                        "start_distance": start_distance,
                        "final_distance": path_distance,
                        "attempts": attempts,
                    }
        except (BridgeError, OSError, asyncio.TimeoutError, ValueError) as exc:
            attempts.append({"phase": "terrain_astar", "error": str(exc)})
        deadline = asyncio.get_running_loop().time() + timeout
        last_position = start
        stuck_pulses = 0

        try:
            while asyncio.get_running_loop().time() < deadline:
                observation = await self.observe()
                current = self.player_position(observation)
                distance = self.distance(current, target)
                if distance <= stop_distance:
                    return {
                        "status": "success",
                        "reached": True,
                        "target_x": x,
                        "target_y": y,
                        "target_z": z,
                        "start_distance": start_distance,
                        "final_distance": distance,
                        "attempts": attempts,
                    }

                eye = (current[0], current[1] + 1.62, current[2])
                flat_target = (target[0], eye[1], target[2])
                yaw, _ = self.yaw_pitch(eye, flat_target)
                moved_since_last = self.distance(current, last_position)
                if moved_since_last < 0.12:
                    stuck_pulses += 1
                else:
                    stuck_pulses = 0
                last_position = current

                cleared_path: List[Dict[str, Any]] = []
                vertical_gap = target[1] - current[1]
                nav_x, nav_y, nav_z = x, y, z
                nav_stop_distance = stop_distance
                nav_timeout = 1.2
                nav_sprint = sprint and distance > 6.0
                if vertical_gap > 1.25:
                    cleared_path = await self.clear_upward_path(observation, yaw)
                    step_x, step_z = self.forward_step(yaw)
                    base_x, base_y, base_z = int(math.floor(current[0])), int(math.floor(current[1])), int(math.floor(current[2]))
                    nav_x, nav_y, nav_z = base_x + step_x, base_y + 1, base_z + step_z
                    nav_stop_distance = 0.85
                    nav_timeout = 2.4
                    nav_sprint = False
                    await self.client.request("jump")
                elif stuck_pulses >= 2:
                    cleared_path = await self.clear_forward_path(observation, yaw)
                    await self.client.request("jump")

                navigation = await self.client.request(
                    "navigate_to",
                    x=nav_x,
                    y=nav_y,
                    z=nav_z,
                    stop_distance=nav_stop_distance,
                    timeout=nav_timeout,
                    sprint=nav_sprint,
                )
                attempt = {
                    "distance": distance,
                    "vertical_gap": vertical_gap,
                    "yaw": yaw,
                    "moved_since_last": moved_since_last,
                    "stuck_pulses": stuck_pulses,
                    "navigation_target": {"x": nav_x, "y": nav_y, "z": nav_z},
                    "navigation": navigation,
                }
                if cleared_path:
                    attempt["cleared_path"] = cleared_path
                attempts.append(attempt)
                await asyncio.sleep(0.8)
        finally:
            await self.stop_all()

        final = self.player_position(await self.observe())
        final_distance = self.distance(final, target)
        return {
            "status": "timeout",
            "reached": final_distance <= stop_distance,
            "target_x": x,
            "target_y": y,
            "target_z": z,
            "start_distance": start_distance,
            "final_distance": final_distance,
            "attempts": attempts,
        }

    async def dig_toward_target(
        self,
        observation: Dict[str, Any],
        target_response: Optional[Dict[str, Any]] = None,
        *,
        max_steps: int = 10,
    ) -> SkillResult:
        start = self.player_position(observation)
        target: Optional[Tuple[float, float, float]] = None
        if target_response and target_response.get("found"):
            target = (
                float(target_response["x"]) + 0.5,
                float(target_response["y"]) + 0.5,
                float(target_response["z"]) + 0.5,
            )
        steps: List[Dict[str, Any]] = []
        reached_resource = False

        for _ in range(max(1, max_steps)):
            observation = await self.observe()
            current = self.player_position(observation)
            yaw = await self.escape_yaw(observation, current, target)
            step_x, step_z = self.forward_step(yaw)
            base_x, base_y, base_z = int(math.floor(current[0])), int(math.floor(current[1])), int(math.floor(current[2]))
            next_step = (base_x + step_x, base_y + 1, base_z + step_z)
            before_distance = self.distance(current, target) if target is not None else None

            cleared = await self.clear_upward_path(observation, yaw)
            await self.client.request("jump")
            navigation = await self.client.request(
                "navigate_to",
                x=next_step[0],
                y=next_step[1],
                z=next_step[2],
                stop_distance=0.85,
                timeout=2.6,
                sprint=False,
            )
            await asyncio.sleep(1.05)
            await self.stop_all()

            refreshed = await self.observe()
            after = self.player_position(refreshed)
            after_distance = self.distance(after, target) if target is not None else None
            try:
                reachable_log = await self.client.request(
                    "find_nearest_block",
                    block_type="log",
                    max_radius=24,
                    reachable_only=True,
                    exposed_only=True,
                    avoid_undermining=True,
                    max_vertical_difference=6,
                    max_reach_candidate_distance=6.0,
                )
                reached_resource = bool(reachable_log.get("found"))
            except BridgeError:
                reachable_log = {"found": False}

            step_details = {
                "from": {"x": current[0], "y": current[1], "z": current[2]},
                "to": {"x": after[0], "y": after[1], "z": after[2]},
                "yaw": yaw,
                "target_step": {"x": next_step[0], "y": next_step[1], "z": next_step[2]},
                "navigation": navigation,
                "cleared": cleared,
                "reachable_log": reachable_log,
            }
            if before_distance is not None and after_distance is not None:
                step_details["distance_delta"] = before_distance - after_distance
            steps.append(step_details)

            gained_height = after[1] - start[1]
            environment = refreshed.get("environment", {})
            if reached_resource or environment.get("can_see_sky") or gained_height >= 4.0:
                final_distance = self.distance(after, target) if target is not None else None
                return SkillResult(
                    "dig_toward_surface",
                    success=True,
                    reward_hint=0.16 + max(0.0, min(0.24, gained_height * 0.04)),
                    details={
                        "start": {"x": start[0], "y": start[1], "z": start[2]},
                        "final": {"x": after[0], "y": after[1], "z": after[2]},
                        "height_delta": gained_height,
                        "final_distance": final_distance,
                        "target": target_response or {},
                        "steps": steps,
                    },
                )

        final_observation = await self.observe()
        final = self.player_position(final_observation)
        height_delta = final[1] - start[1]
        final_distance = self.distance(final, target) if target is not None else None
        start_distance = self.distance(start, target) if target is not None else None
        distance_delta = (start_distance - final_distance) if start_distance is not None and final_distance is not None else 0.0
        success = height_delta >= 1.0 or distance_delta >= 2.0 or bool(final_observation.get("environment", {}).get("can_see_sky"))
        return SkillResult(
            "dig_toward_surface",
            success=success,
            reward_hint=0.08 if success else -0.04,
            details={
                "start": {"x": start[0], "y": start[1], "z": start[2]},
                "final": {"x": final[0], "y": final[1], "z": final[2]},
                "height_delta": height_delta,
                "distance_delta": distance_delta,
                "final_distance": final_distance,
                "target": target_response or {},
                "steps": steps,
            },
        )

    async def escape_yaw(
        self,
        observation: Dict[str, Any],
        current: Tuple[float, float, float],
        target: Optional[Tuple[float, float, float]],
    ) -> float:
        if target is not None:
            eye = (current[0], current[1] + 1.62, current[2])
            flat_target = (target[0], eye[1], target[2])
            yaw, _ = self.yaw_pitch(eye, flat_target)
            return yaw

        player = observation.get("player", {})
        fallback_yaw = float(player.get("yaw", 0.0))
        try:
            payload = await self.client.request("get_visual_summary", rays=9, distance=48.0)
        except BridgeError:
            return fallback_yaw
        summary = payload.get("visual_summary", {}) if isinstance(payload.get("visual_summary"), dict) else {}
        rays = summary.get("openings", []) if isinstance(summary.get("openings"), list) else []
        openings = [
            ray for ray in rays
            if not ray.get("hit") or float(ray.get("hit_distance", 0.0)) >= 6.0
        ]
        if openings:
            opening = max(openings, key=lambda ray: float(ray.get("hit_distance", 0.0)))
            return float(opening.get("yaw", fallback_yaw))
        return fallback_yaw

    async def clear_upward_path(self, observation: Dict[str, Any], yaw: float) -> List[Dict[str, Any]]:
        x, y, z = self.player_position(observation)
        step_x, step_z = self.forward_step(yaw)
        base_x, base_y, base_z = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        columns = [(base_x + step_x, base_z + step_z)]
        if step_x and step_z:
            columns.extend(((base_x + step_x, base_z), (base_x, base_z + step_z)))

        targets: List[Tuple[int, int, int]] = [
            (base_x, base_y + 1, base_z),
            (base_x, base_y + 2, base_z),
        ]
        for column_x, column_z in columns:
            floor_info = await self.client.request("get_block_info", x=column_x, y=base_y, z=column_z)
            heights = (base_y, base_y + 1, base_y + 2) if floor_info.get("is_air") else (base_y + 1, base_y + 2, base_y + 3)
            targets.extend((column_x, block_y, column_z) for block_y in heights)

        second_x, second_z = base_x + step_x * 2, base_z + step_z * 2
        targets.extend(((second_x, base_y + 2, second_z), (second_x, base_y + 3, second_z)))
        unique_targets = list(dict.fromkeys(targets))
        return [await self.mine_block_until_changed(observation, block, attempts=42, delay=0.1) for block in unique_targets]

    async def clear_forward_path(self, observation: Dict[str, Any], yaw: float) -> List[Dict[str, Any]]:
        x, y, z = self.player_position(observation)
        step_x, step_z = self.forward_step(yaw)
        base_x, base_y, base_z = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        columns = [(base_x + step_x, base_z + step_z)]
        if step_x and step_z:
            columns.extend(((base_x + step_x, base_z), (base_x, base_z + step_z)))
        targets = tuple(
            (column_x, block_y, column_z)
            for column_x, column_z in columns
            for block_y in (base_y, base_y + 1, base_y + 2)
        )
        return [await self.mine_block_until_changed(observation, block, attempts=36, delay=0.1) for block in targets]

    async def mine_block_until_changed(
        self,
        observation: Dict[str, Any],
        block: Tuple[int, int, int],
        *,
        attempts: int,
        delay: float,
    ) -> Dict[str, Any]:
        try:
            info = await self.client.request("get_block_info", x=block[0], y=block[1], z=block[2])
        except BridgeError as exc:
            return {"x": block[0], "y": block[1], "z": block[2], "error": str(exc)}

        block_id = str(info.get("block", ""))
        details: Dict[str, Any] = {"x": block[0], "y": block[1], "z": block[2], "from": block_id}
        if info.get("is_air"):
            details.update({"skipped": "air", "to": block_id})
            return details
        if any(term in block_id for term in DANGEROUS_BLOCK_TERMS):
            details.update({"skipped": "dangerous", "to": block_id})
            return details
        if any(term in block_id for term in FLUID_BLOCK_TERMS):
            details.update({"skipped": "fluid", "to": block_id})
            return details
        if any(term in block_id for term in UNBREAKABLE_BLOCK_TERMS):
            details.update({"skipped": "unbreakable", "to": block_id})
            return details

        await self.look_at_block(observation, block)
        try:
            details["tool"] = await self.client.request("select_best_tool", x=block[0], y=block[1], z=block[2])
        except BridgeError as exc:
            details["tool_error"] = str(exc)

        final: Dict[str, Any] = {}
        for _ in range(max(1, attempts)):
            try:
                await self.client.request("mine", x=block[0], y=block[1], z=block[2])
            except BridgeError as exc:
                details["mine_error"] = str(exc)
                break
            await asyncio.sleep(delay)
            final = await self.client.request("get_block_info", x=block[0], y=block[1], z=block[2])
            if final.get("is_air") or str(final.get("block", "")) != block_id:
                break
        details["to"] = final.get("block", block_id) if final else block_id
        details["cleared"] = bool(final.get("is_air")) if final else False
        if final and str(final.get("block", "")) != block_id:
            details["changed"] = True
        return details

    async def look_at_block(self, observation: Dict[str, Any], block: Tuple[int, int, int]) -> None:
        pos = self.player_position(observation)
        eye = (pos[0], pos[1] + 1.62, pos[2])
        target = (block[0] + 0.5, block[1] + 0.5, block[2] + 0.5)
        yaw, pitch = self.yaw_pitch(eye, target)
        await self.client.request("look", yaw=yaw, pitch=pitch)

    @staticmethod
    def yaw_pitch(origin: Tuple[float, float, float], target: Tuple[float, float, float]) -> Tuple[float, float]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        dz = target[2] - origin[2]
        horizontal = math.sqrt(dx * dx + dz * dz)
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = math.degrees(math.atan2(-dy, horizontal))
        return yaw, max(-90.0, min(90.0, pitch))

    @staticmethod
    def forward_step(yaw: float) -> Tuple[int, int]:
        radians = math.radians(yaw)
        dx = int(round(-math.sin(radians)))
        dz = int(round(math.cos(radians)))
        if dx == 0 and dz == 0:
            return 0, 1
        return dx, dz

    @staticmethod
    def should_eat(observation: Dict[str, Any]) -> bool:
        player = observation.get("player", {})
        survival = observation.get("survival", {})
        health = float(player.get("health", 20.0))
        food = float(player.get("food", 20.0))
        return bool(survival.get("should_eat")) or food <= 14.0 or (health <= 10.0 and food < 20.0)

    @staticmethod
    def in_environmental_danger(observation: Dict[str, Any]) -> bool:
        player = observation.get("player", {})
        if player.get("is_in_lava") or player.get("is_on_fire"):
            return True
        if float(player.get("fall_distance", 0.0)) > 7.0:
            return True
        nearby_blocks = observation.get("nearby_blocks", [])
        return any(
            bool(block.get("hazard")) or any(term in str(block.get("block", "")) for term in DANGEROUS_BLOCK_TERMS)
            for block in nearby_blocks
            if float(block.get("distance", 99.0)) <= 2.5
        )

    @staticmethod
    def low_oxygen(player: Dict[str, Any]) -> bool:
        air = float(player.get("air", player.get("max_air", 300)))
        max_air = max(1.0, float(player.get("max_air", 300)))
        return bool(player.get("is_underwater")) and air / max_air < 0.45

    @staticmethod
    def visible_resource_nearby(observation: Dict[str, Any]) -> bool:
        blocks = observation.get("nearby_blocks", [])
        return any(
            bool(block.get("resource")) or any(term in str(block.get("block", "")) for term in RESOURCE_TERMS)
            for block in blocks
            if float(block.get("distance", 99.0)) <= 12.0
        )

    @staticmethod
    def inventory_item_count(inventory: Dict[str, Any], terms: Iterable[str]) -> int:
        normalized_terms = tuple(str(term).lower() for term in terms)
        total = 0
        for item in inventory.get("items", []):
            item_id = str(item.get("item", "")).lower()
            if any(term in item_id for term in normalized_terms):
                total += int(item.get("count", 0))
        return total

    @staticmethod
    def player_position(observation: Dict[str, Any]) -> Tuple[float, float, float]:
        position = observation.get("player", {}).get("position", {})
        return (float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0)))

    @staticmethod
    def distance(a: Iterable[float], b: Iterable[float]) -> float:
        ax, ay, az = a
        bx, by, bz = b
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    @staticmethod
    def is_hostile(entity: Dict[str, Any]) -> bool:
        entity_type = str(entity.get("type", ""))
        return any(keyword in entity_type for keyword in HOSTILE_KEYWORDS)

    @staticmethod
    def closest_entity(observation: Dict[str, Any], predicate: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = observation.get("nearby_entities", [])
        candidates = [entity for entity in entities if predicate(entity)]
        if not candidates:
            return None
        return min(candidates, key=lambda entity: float(entity.get("distance", 999999.0)))
