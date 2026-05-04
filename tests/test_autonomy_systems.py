import math
import os
import sys
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import BridgeClient
from combat import CombatTactics
from mechanics import BasicMechanics
from mechanics import SkillResult
from navigation import TerrainAStar
from perception import WorldMemory, build_world_state
from planning import UtilityPlanner
from validation import MultiSeedResult, SeedRunMetrics


def observation(
    *,
    health=20.0,
    food=20,
    inventory=None,
    blocks=None,
    entities=None,
    danger_level="none",
):
    return {
        "player": {
            "health": health,
            "max_health": 20.0,
            "food": food,
            "saturation": 3.0,
            "position": {"x": 0.5, "y": 64.0, "z": 0.5},
            "yaw": 0.0,
            "pitch": 0.0,
            "is_on_ground": True,
            "is_dead": health <= 0,
            "armor_value": 0,
        },
        "inventory": {"items": inventory or []},
        "environment": {"is_day": True, "combined_light": 15},
        "nearby_blocks": blocks or [],
        "nearby_entities": entities or [],
        "threats": {"danger_level": danger_level, "score": 0.0, "entities": [], "hazards": []},
        "survival": {"should_eat": food <= 14},
        "combat": {"weapon_score": 2.0, "attack_strength": 1.0},
        "vision": {},
    }


def ground(radius=4):
    blocks = []
    for x in range(-radius, radius + 1):
        for z in range(-radius, radius + 1):
            blocks.append({"x": x, "y": 63, "z": z, "block": "minecraft:grass_block", "distance": math.hypot(x, z)})
    return blocks


class WorldStateTests(unittest.TestCase):
    def test_classifies_resources_hazards_and_threats(self):
        obs = observation(
            blocks=[
                {"x": 2, "y": 64, "z": 0, "block": "minecraft:oak_log", "distance": 2.0, "resource": True},
                {"x": 1, "y": 64, "z": 0, "block": "minecraft:lava", "distance": 1.0, "hazard": True},
            ],
            entities=[
                {"id": 7, "type": "minecraft:zombie", "x": 3.0, "y": 64.0, "z": 0.0, "distance": 3.0, "hostile": True},
                {"id": 8, "type": "minecraft:item", "item_name": "minecraft:apple", "item_count": 1, "x": 1.0, "y": 64.0, "z": 2.0, "distance": 2.2},
            ],
        )
        world = build_world_state(obs)
        self.assertTrue(world.nearest_resource("log"))
        self.assertEqual(world.hostile_count(), 1)
        self.assertIn((1, 64, 0), world.terrain.hazards)
        self.assertTrue(any(resource.source == "item" for resource in world.resources))


class UtilityPlannerTests(unittest.TestCase):
    def test_eating_beats_progression_when_starving(self):
        obs = observation(
            food=5,
            inventory=[{"item": "minecraft:bread", "count": 1, "is_edible": True}],
            blocks=ground(),
        )
        world = build_world_state(obs)
        decision = UtilityPlanner("survive and progress").decide(world, WorldMemory())
        self.assertEqual(decision.skill, "eat_food")

    def test_crafting_planks_is_selected_from_logs(self):
        obs = observation(
            inventory=[{"item": "minecraft:oak_log", "count": 2}],
            blocks=ground(),
        )
        world = build_world_state(obs)
        decision = UtilityPlanner("survive and progress").decide(world, WorldMemory())
        self.assertEqual(decision.skill, "craft_planks")

    def test_combat_uses_adaptive_layer(self):
        obs = observation(
            blocks=ground(),
            entities=[{"id": 4, "type": "minecraft:zombie", "x": 2.0, "y": 64.0, "z": 0.0, "distance": 2.0, "hostile": True, "threat_score": 12.0}],
            danger_level="high",
        )
        world = build_world_state(obs)
        decision = UtilityPlanner("survive").decide(world, WorldMemory())
        self.assertEqual(decision.skill, "adaptive_combat")

    def test_utility_does_not_chase_low_value_junk_drops(self):
        obs = observation(
            blocks=ground(),
            entities=[
                {
                    "id": 9,
                    "type": "minecraft:item",
                    "item_name": "minecraft:dirt",
                    "item_count": 1,
                    "x": 1.0,
                    "y": 64.0,
                    "z": 0.0,
                    "distance": 1.0,
                }
            ],
        )
        world = build_world_state(obs)
        decision = UtilityPlanner("survive and progress").decide(world, WorldMemory())
        self.assertNotEqual(decision.skill, "collect_visible_item")
        self.assertFalse(any(option["skill"] == "collect_visible_item" for option in decision.alternatives))

    def test_recovery_beats_mining_after_repeated_failures(self):
        obs = observation(
            inventory=[
                {"item": "minecraft:spruce_planks", "count": 8},
                {"item": "minecraft:stick", "count": 4},
                {"item": "minecraft:crafting_table", "count": 1},
                {"item": "minecraft:wooden_pickaxe", "count": 1},
            ],
            blocks=ground(),
        )
        world = build_world_state(obs)
        memory = WorldMemory()
        for _ in range(3):
            memory.record_result("mine_nearest_stone", False)
        decision = UtilityPlanner("survive and progress").decide(world, memory, recent_failures=3)
        self.assertEqual(decision.category, "recovery")


class TerrainAStarTests(unittest.TestCase):
    def test_astar_routes_around_hazard(self):
        blocks = ground()
        blocks.append({"x": 1, "y": 64, "z": 0, "block": "minecraft:lava", "distance": 1.0, "hazard": True})
        world = build_world_state(observation(blocks=blocks))
        plan = TerrainAStar(world).plan((0, 64, 0), (3, 64, 0), max_nodes=512)
        self.assertTrue(plan.path)
        self.assertNotIn((1, 64, 0), plan.path)

    def test_astar_rejects_unsafe_fall(self):
        world = build_world_state(observation(blocks=ground()))
        astar = TerrainAStar(world, max_safe_fall=3)
        feasible, cost = astar.feasible_step((0, 64, 0), (1, 60, 0), False)
        self.assertFalse(feasible)
        self.assertTrue(math.isinf(cost))

    def test_astar_allows_safe_drop_with_penalty(self):
        world = build_world_state(observation(blocks=ground()))
        astar = TerrainAStar(world, max_safe_fall=3)
        feasible, cost = astar.feasible_step((0, 64, 0), (1, 61, 0), False)
        self.assertTrue(feasible)
        self.assertGreater(cost, 1.0)


class CombatTacticsTests(unittest.TestCase):
    def test_overwhelmed_combat_flees(self):
        entities = [
            {"id": index, "type": "minecraft:zombie", "x": float(index), "y": 64.0, "z": 1.0, "distance": 2.0 + index * 0.2, "hostile": True, "threat_score": 10.0}
            for index in range(6)
        ]
        world = build_world_state(observation(health=8, blocks=ground(), entities=entities, danger_level="high"))
        decision = CombatTactics().decide(world)
        self.assertEqual(decision.strategy, "flee")
        self.assertIsNotNone(decision.escape_path)

    def test_single_favorable_mob_engages(self):
        world = build_world_state(
            observation(
                health=20,
                blocks=ground(),
                entities=[{"id": 1, "type": "minecraft:zombie", "x": 3.0, "y": 64.0, "z": 0.0, "distance": 3.0, "hostile": True, "threat_score": 5.0}],
            )
        )
        decision = CombatTactics().decide(world)
        self.assertEqual(decision.strategy, "engage")

    def test_environmental_danger_flees_without_target(self):
        obs = observation(
            blocks=ground(),
            danger_level="moderate",
        )
        obs["threats"]["hazards"] = [
            {"x": 1, "y": 64, "z": 0, "block": "minecraft:lava", "distance": 1.0, "hazard": True}
        ]
        world = build_world_state(obs)
        decision = CombatTactics().decide(world)
        self.assertEqual(decision.strategy, "flee")
        self.assertIsNone(decision.target)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def request(self, action: str, **payload: Any) -> Dict[str, Any]:
        self.calls.append((action, payload))
        if action == "get_threats":
            return {"threats": {"danger_level": "none"}}
        if action == "get_vision":
            return {"vision": {}}
        return {"status": "success"}


class MechanicsAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_adaptive_combat_flees_environmental_hazard_without_target(self):
        obs = observation(blocks=ground(), danger_level="moderate")
        obs["threats"]["hazards"] = [
            {"x": 1, "y": 64, "z": 0, "block": "minecraft:lava", "distance": 1.0, "hazard": True}
        ]
        client = FakeClient()
        mechanics = BasicMechanics(client, step_delay=0.0)
        result = await mechanics.skill_adaptive_combat(obs)
        self.assertTrue(result.success)
        self.assertTrue(any(action in {"flee", "navigate_to"} for action, _payload in client.calls))

    async def test_mine_nearest_stone_does_not_stop_at_one_cobblestone(self):
        client = FakeClient()

        async def request(action: str, **payload: Any) -> Dict[str, Any]:
            client.calls.append((action, payload))
            if action == "find_nearest_block":
                return {
                    "found": True,
                    "x": 1,
                    "y": 64,
                    "z": 0,
                    "block": "minecraft:stone",
                    "distance": 1.0,
                    "reach_distance": 3.0,
                }
            return {"status": "success"}

        client.request = request
        mechanics = BasicMechanics(client, step_delay=0.0)

        async def mine_known_block(*_args: Any, **_kwargs: Any) -> SkillResult:
            return SkillResult("mine_nearest_stone", reward_hint=0.45, details={"mined_more": True})

        mechanics.mine_known_block = mine_known_block  # type: ignore[method-assign]
        obs = observation(
            inventory=[{"item": "minecraft:cobblestone", "count": 1}],
            blocks=ground(),
        )
        result = await mechanics.skill_mine_nearest_stone(obs)
        self.assertTrue(result.success)
        self.assertEqual(result.details, {"mined_more": True})

    async def test_mine_nearest_stone_exposes_buried_stone(self):
        client = FakeClient()
        obs = observation(blocks=ground())
        mined_target = False

        async def request(action: str, **payload: Any) -> Dict[str, Any]:
            nonlocal mined_target
            client.calls.append((action, payload))
            if action == "get_full_state":
                return obs
            if action == "get_inventory":
                return {"inventory": {"items": []}}
            if action == "find_nearest_block":
                if payload.get("exposed_only"):
                    return {"found": False}
                return {
                    "found": True,
                    "x": 0,
                    "y": 62,
                    "z": 0,
                    "block": "minecraft:stone",
                    "distance": 2.5,
                    "reach_distance": 4.0,
                }
            if action == "get_block_info":
                y = int(payload.get("y", 0))
                if y == 62:
                    return {"block": "minecraft:air" if mined_target else "minecraft:stone", "is_air": mined_target}
                return {"block": "minecraft:air", "is_air": True}
            if action == "mine" and int(payload.get("y", 0)) == 62:
                mined_target = True
            return {"status": "success"}

        client.request = request
        mechanics = BasicMechanics(client, step_delay=0.0)
        result = await mechanics.skill_mine_nearest_stone(obs, desired_count=1)
        self.assertTrue(result.success)
        self.assertEqual(result.details["phase"], "expose_buried_stone")
        self.assertTrue(any(action == "mine" and payload.get("y") == 62 for action, payload in client.calls))

    async def test_collect_visible_item_records_visible_skill_name(self):
        client = FakeClient()
        mechanics = BasicMechanics(client, step_delay=0.0)

        async def collect_item(_observation: Dict[str, Any], item_terms=None) -> SkillResult:
            return SkillResult("collect_item", success=False, reward_hint=-0.03, details={"item_terms": item_terms})

        mechanics.skill_collect_item = collect_item  # type: ignore[method-assign]
        result = await mechanics.skill_collect_visible_item(observation(blocks=ground()))
        self.assertFalse(result.success)
        self.assertEqual(result.name, "collect_visible_item")
        self.assertEqual(result.details["delegated_skill"], "collect_item")

    async def test_mine_block_skips_water_without_mining(self):
        client = FakeClient()

        async def request(action: str, **payload: Any) -> Dict[str, Any]:
            client.calls.append((action, payload))
            if action == "get_block_info":
                return {"block": "minecraft:water", "is_air": False}
            return {"status": "success"}

        client.request = request
        mechanics = BasicMechanics(client, step_delay=0.0)
        result = await mechanics.mine_block_until_changed(observation(blocks=ground()), (1, 64, 0), attempts=3, delay=0.0)
        self.assertEqual(result["skipped"], "fluid")
        self.assertFalse(any(action == "mine" for action, _payload in client.calls))


class BridgeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_request_id_before_fifo(self):
        client = BridgeClient()
        loop = __import__("asyncio").get_running_loop()
        request_future = loop.create_future()
        fifo_future = loop.create_future()
        client._pending["abc"] = request_future
        client._pending_fifo.append(fifo_future)
        client._route_message({"request_id": "abc", "status": "success", "value": 7})
        self.assertTrue(request_future.done())
        self.assertEqual(request_future.result()["value"], 7)
        self.assertFalse(fifo_future.done())


class ValidationTests(unittest.TestCase):
    def test_multiseed_summary_reports_survival_and_milestones(self):
        result = MultiSeedResult(
            [
                SeedRunMetrics(seed="1", world="AISeed_1", survived=True, final_stage=4, time_to_wood=10.0),
                SeedRunMetrics(seed="2", world="AISeed_2", survived=False, final_stage=2, failures=["death"]),
            ]
        )
        summary = result.summary()
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["survival_rate"], 0.5)
        self.assertEqual(summary["average_time_to_wood"], 10.0)
        self.assertEqual(summary["failures"], ["death"])


if __name__ == "__main__":
    unittest.main()
