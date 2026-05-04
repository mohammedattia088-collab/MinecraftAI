import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


WOOD_TERMS = ("log", "wood", "planks", "stick", "crafting_table")
STONE_TERMS = ("cobblestone", "cobbled_deepslate")
PICKAXE_TERMS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
DIAMOND_TERMS = ("diamond", "diamond_ore", "deepslate_diamond_ore")
IRON_TERMS = ("iron_ingot", "raw_iron", "iron_ore", "deepslate_iron_ore")


@dataclass(frozen=True)
class ObjectiveDecision:
    skill: str
    reason: str
    objective: str
    priority: float = 1.0


class ObjectivePlanner:
    """
    Methodological planner for Minecraft progression.

    It keeps the learner from asking the policy to solve impossible goals in one
    leap. Hard objectives are decomposed into prerequisites: safety, tools,
    materials, depth, then the target itself.
    """

    def __init__(self, goal: str = "survive and progress") -> None:
        self.goal = self.normalize_goal(goal)

    def select_skill(self, observation: Dict[str, Any]) -> Optional[ObjectiveDecision]:
        player = observation.get("player", {})
        survival = observation.get("survival", {})
        if survival.get("should_eat"):
            return ObjectiveDecision("eat_food", "survival_food", "stay_alive", 10.0)
        if player.get("is_in_lava") or player.get("is_on_fire"):
            return ObjectiveDecision("flee_hostile", "environmental_danger", "stay_alive", 10.0)

        if "diamond" in self.goal:
            return self.diamond_plan(observation)
        if any(term in self.goal for term in ("achievement", "advancement", "progress")):
            return self.advancement_plan(observation)
        if "wood" in self.goal or "tree" in self.goal:
            return ObjectiveDecision(random.choice(("scan_logs", "mine_nearest_log", "harvest_trees")), "wood_goal", "collect_wood", 3.0)
        if "explore" in self.goal:
            return ObjectiveDecision(random.choice(("visual_scan", "sprint_wander", "scan_environment")), "explore_goal", "map_world", 2.0)
        return self.advancement_plan(observation)

    def diamond_plan(self, observation: Dict[str, Any]) -> ObjectiveDecision:
        inventory = observation.get("inventory", {})
        y = float(observation.get("player", {}).get("position", {}).get("y", 64.0))

        if self.count_items(inventory, DIAMOND_TERMS) > 0:
            return ObjectiveDecision("advancement_status", "diamond_acquired", "lock_in_progress", 4.0)
        if self.nearby_block_matches(observation, DIAMOND_TERMS):
            return ObjectiveDecision("mine_nearest_diamond", "diamond_visible", "mine_diamond", 9.0)

        prerequisite = self.early_tool_prerequisite(observation)
        if prerequisite is not None:
            return prerequisite

        if self.count_items(inventory, ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")) <= 0:
            return ObjectiveDecision("craft_stone_pickaxe", "need_pickaxe_for_diamonds", "tool_up", 7.0)
        if y > -46:
            return ObjectiveDecision("descend_to_diamond_layer", "need_diamond_depth", "reach_y_minus_54", 6.0)
        return ObjectiveDecision(random.choice(("mine_nearest_diamond", "branch_mine", "visual_scan")), "diamond_branch_search", "find_diamonds", 5.0)

    def advancement_plan(self, observation: Dict[str, Any]) -> ObjectiveDecision:
        prerequisite = self.early_tool_prerequisite(observation)
        if prerequisite is not None:
            return prerequisite

        inventory = observation.get("inventory", {})
        if self.count_items(inventory, IRON_TERMS) <= 0 and self.nearby_block_matches(observation, ("iron_ore", "deepslate_iron_ore")):
            return ObjectiveDecision("mine_nearest_resource", "visible_iron", "acquire_iron", 5.0)
        if self.count_items(inventory, ("furnace",)) <= 0 and self.count_items(inventory, STONE_TERMS) >= 8:
            return ObjectiveDecision("craft_furnace", "need_furnace", "smelt_iron", 4.5)
        if self.nearby_block_matches(observation, DIAMOND_TERMS):
            return ObjectiveDecision("mine_nearest_diamond", "visible_diamond", "story_mine_diamond", 6.0)
        if self.closest_hostile_nearby(observation):
            return ObjectiveDecision("engage_hostile", "combat_advancement_opportunity", "combat_curriculum", 3.0)
        if self.visible_resource_nearby(observation):
            return ObjectiveDecision("mine_nearest_resource", "resource_curriculum", "collect_resources", 2.5)
        return ObjectiveDecision(random.choice(("visual_scan", "sprint_wander", "scan_environment")), "curriculum_explore", "discover_next_task", 1.5)

    def early_tool_prerequisite(self, observation: Dict[str, Any]) -> Optional[ObjectiveDecision]:
        inventory = observation.get("inventory", {})
        if self.count_items(inventory, ("log", "stem", "hyphae", "wood", "planks")) <= 0:
            return ObjectiveDecision(random.choice(("scan_logs", "mine_nearest_log", "harvest_trees")), "need_wood", "collect_wood", 8.0)
        if self.count_items(inventory, ("planks",)) <= 0:
            return ObjectiveDecision("craft_planks", "need_planks", "craft_planks", 7.5)
        if self.count_items(inventory, ("crafting_table",)) <= 0:
            return ObjectiveDecision("craft_crafting_table", "need_table", "craft_table", 7.0)
        if self.count_items(inventory, ("stick",)) < 2:
            return ObjectiveDecision("craft_sticks", "need_sticks", "craft_sticks", 6.5)
        if self.count_items(inventory, PICKAXE_TERMS) <= 0:
            return ObjectiveDecision("craft_wooden_pickaxe", "need_first_pickaxe", "craft_wooden_pickaxe", 6.0)
        if self.count_items(inventory, STONE_TERMS) < 3 and self.count_items(inventory, ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")) <= 0:
            return ObjectiveDecision("mine_nearest_stone", "need_cobblestone", "mine_stone", 5.5)
        return None

    def progression_reward(self, before: Dict[str, Any], after: Dict[str, Any], result_name: str) -> float:
        before_stage = self.stage_score(before)
        after_stage = self.stage_score(after)
        reward = max(0, after_stage - before_stage) * 0.85
        if result_name.startswith("craft_"):
            reward += 0.06
        return reward

    def state_bucket(self, observation: Dict[str, Any]) -> str:
        inventory = observation.get("inventory", {})
        y = float(observation.get("player", {}).get("position", {}).get("y", 64.0))
        stage = self.stage_score(observation)
        depth = "diamond_depth" if y <= -46 else "underground" if y < 45 else "surface"
        diamonds = "has_diamond" if self.count_items(inventory, DIAMOND_TERMS) > 0 else "no_diamond"
        return f"stage_{stage}|{depth}|{diamonds}"

    def stage_score(self, observation: Dict[str, Any]) -> int:
        inventory = observation.get("inventory", {})
        score = 0
        if self.count_items(inventory, ("log", "planks", "stick", "crafting_table")) > 0:
            score += 1
        if self.count_items(inventory, ("planks",)) > 0:
            score += 1
        if self.count_items(inventory, ("crafting_table",)) > 0:
            score += 1
        if self.count_items(inventory, ("stick",)) >= 2:
            score += 1
        if self.count_items(inventory, PICKAXE_TERMS) > 0:
            score += 1
        if self.count_items(inventory, STONE_TERMS) >= 3:
            score += 1
        if self.count_items(inventory, ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")) > 0:
            score += 1
        if self.count_items(inventory, IRON_TERMS) > 0:
            score += 1
        if self.count_items(inventory, DIAMOND_TERMS) > 0:
            score += 2
        return score

    @staticmethod
    def count_items(inventory: Dict[str, Any], terms: Iterable[str]) -> int:
        normalized = tuple(str(term).lower() for term in terms)
        total = 0
        for item in inventory.get("items", []):
            item_id = str(item.get("item", "")).lower()
            if any(term in item_id for term in normalized):
                total += int(item.get("count", 0))
        return total

    @staticmethod
    def nearby_block_matches(observation: Dict[str, Any], terms: Iterable[str]) -> bool:
        normalized = tuple(str(term).lower() for term in terms)
        return any(
            any(term in str(block.get("block", "")).lower() for term in normalized)
            for block in observation.get("nearby_blocks", [])
        )

    @staticmethod
    def visible_resource_nearby(observation: Dict[str, Any]) -> bool:
        return any(
            bool(block.get("resource"))
            for block in observation.get("nearby_blocks", [])
            if float(block.get("distance", 99.0)) <= 14.0
        )

    @staticmethod
    def closest_hostile_nearby(observation: Dict[str, Any]) -> bool:
        hostile_terms = ("zombie", "skeleton", "creeper", "spider", "witch", "pillager", "warden", "blaze")
        return any(
            any(term in str(entity.get("type", "")).lower() for term in hostile_terms)
            for entity in observation.get("nearby_entities", [])
        )

    @staticmethod
    def normalize_goal(goal: str) -> str:
        return " ".join(str(goal or "").strip().lower().replace("_", " ").split())
