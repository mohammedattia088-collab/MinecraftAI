from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    from ..combat import CombatTactics
    from ..perception.world_state import WorldMemory, WorldState
    from .objectives import ObjectivePlanner
except ImportError:
    from combat import CombatTactics
    from perception.world_state import WorldMemory, WorldState
    from planning.objectives import ObjectivePlanner


@dataclass(frozen=True)
class UtilityAction:
    name: str
    skill: str
    category: str
    reward: Callable[[WorldState, WorldMemory], float]
    cost: Callable[[WorldState, WorldMemory], float]
    risk: Callable[[WorldState, WorldMemory], float]
    precondition: Callable[[WorldState, WorldMemory], bool] = lambda _world, _memory: True

    def evaluate(self, world: WorldState, memory: WorldMemory) -> Optional["UtilityDecision"]:
        if not self.precondition(world, memory):
            return None
        reward = float(self.reward(world, memory))
        cost = float(self.cost(world, memory))
        risk = float(self.risk(world, memory))
        score = reward - cost - risk - memory.failure_penalty(self.skill)
        return UtilityDecision(
            skill=self.skill,
            action=self.name,
            category=self.category,
            score=score,
            reward=reward,
            cost=cost,
            risk=risk,
            reason=f"{self.name}: reward={reward:.2f} cost={cost:.2f} risk={risk:.2f}",
        )


@dataclass(frozen=True)
class UtilityDecision:
    skill: str
    action: str
    category: str
    score: float
    reward: float
    cost: float
    risk: float
    reason: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


class UtilityPlanner:
    """
    Utility-based brain over stable skills.

    Every action exposes reward, cost, risk, and preconditions. The runner can
    still learn inside this structure, but progression no longer depends on one
    fixed script.
    """

    def __init__(self, goal: str = "survive and progress") -> None:
        self.goal = self.normalize_goal(goal)
        self.objectives = ObjectivePlanner(goal)
        self.combat = CombatTactics()
        self.actions: List[UtilityAction] = self._build_actions()

    def decide(
        self,
        world: WorldState,
        memory: WorldMemory,
        *,
        stuck: bool = False,
        recent_failures: int = 0,
    ) -> UtilityDecision:
        decisions = [
            decision for action in self.actions
            if (decision := action.evaluate(world, memory)) is not None
        ]

        objective = self.objectives.select_skill(world.raw)
        if objective is not None:
            objective_score = objective.priority + self.goal_alignment(world, objective.skill) - memory.failure_penalty(objective.skill)
            decisions.append(
                UtilityDecision(
                    skill=objective.skill,
                    action=f"objective:{objective.objective}",
                    category="objective",
                    score=objective_score,
                    reward=objective.priority + self.goal_alignment(world, objective.skill),
                    cost=memory.failure_penalty(objective.skill),
                    risk=self.world_risk(world) * 0.25,
                    reason=f"{objective.objective}:{objective.reason}",
                )
            )

        if stuck:
            decisions.extend(self.unstuck_decisions(world, memory))
        if recent_failures >= 3:
            decisions.append(
                UtilityDecision(
                    skill=random.choice(("scan_environment", "visual_scan", "look_around")),
                    action="recover_from_failures",
                    category="recovery",
                    score=7.0 + min(2.0, (recent_failures - 3) * 0.4),
                    reward=7.4,
                    cost=0.2,
                    risk=0.0,
                    reason="recent_failures",
                )
            )

        if not decisions:
            return UtilityDecision("visual_scan", "default_scan", "exploration", 0.0, 0.1, 0.1, 0.0, "no_viable_actions")

        decisions.sort(key=lambda decision: (decision.score, random.random()), reverse=True)
        best = decisions[0]
        alternatives = [
            {
                "skill": decision.skill,
                "action": decision.action,
                "category": decision.category,
                "score": round(decision.score, 3),
            }
            for decision in decisions[1:6]
        ]
        return UtilityDecision(
            skill=best.skill,
            action=best.action,
            category=best.category,
            score=best.score,
            reward=best.reward,
            cost=best.cost,
            risk=best.risk,
            reason=best.reason,
            alternatives=alternatives,
        )

    def _build_actions(self) -> List[UtilityAction]:
        return [
            UtilityAction(
                "respawn_if_dead",
                "respawn",
                "survival",
                lambda world, _memory: 100.0 if not world.alive else 0.0,
                lambda _world, _memory: 0.1,
                lambda _world, _memory: 0.0,
                lambda world, _memory: not world.alive,
            ),
            UtilityAction(
                "eat_when_needed",
                "eat_food",
                "eating",
                lambda world, _memory: 9.0 + max(0.0, 14.0 - world.food) + max(0.0, 12.0 - world.health) * 0.4,
                lambda world, _memory: 0.4 if world.edible_count() > 0 else 6.0,
                lambda world, _memory: self.world_risk(world) * 0.15,
                lambda world, _memory: world.hungry,
            ),
            UtilityAction(
                "escape_environment",
                "flee_hostile",
                "survival",
                lambda world, _memory: 18.0 if world.in_environmental_danger else 0.0,
                lambda _world, _memory: 0.8,
                lambda world, _memory: self.world_risk(world) * 0.1,
                lambda world, _memory: world.in_environmental_danger,
            ),
            UtilityAction(
                "adaptive_combat",
                "adaptive_combat",
                "combat",
                lambda world, _memory: self.combat_reward(world),
                lambda world, _memory: 1.2 + max(0.0, 8.0 - world.health) * 0.35,
                lambda world, _memory: self.world_risk(world),
                lambda world, _memory: world.hostile_count(18.0) > 0,
            ),
            UtilityAction(
                "collect_nearby_item",
                "collect_visible_item",
                "mining",
                lambda world, _memory: 1.2 + max((resource.score for resource in world.resources if resource.source == "item"), default=0.0),
                lambda world, _memory: min((resource.distance * 0.25 + max(0.0, 2.0 - resource.score) * 0.5 for resource in world.resources if resource.source == "item"), default=4.0),
                lambda world, _memory: self.world_risk(world) * 0.35,
                lambda world, _memory: any(
                    resource.source == "item" and resource.distance <= 8.0 and resource.score >= 2.0
                    for resource in world.resources
                ),
            ),
            UtilityAction(
                "mine_visible_resource",
                "mine_nearest_resource",
                "mining",
                lambda world, _memory: max((resource.score for resource in world.resources if resource.source == "block"), default=0.0) + self.goal_resource_bonus(world),
                lambda world, _memory: min((max(0.5, resource.distance * 0.18) for resource in world.resources if resource.source == "block"), default=5.0),
                lambda world, _memory: self.world_risk(world) * 0.45,
                lambda world, _memory: any(resource.source == "block" and resource.distance <= 24.0 for resource in world.resources),
            ),
            UtilityAction(
                "seek_wood",
                "mine_nearest_log",
                "mining",
                lambda world, _memory: 8.0 if world.item_count("log", "planks", "wood") <= 0 else 2.0,
                lambda world, _memory: 2.0 if world.nearest_resource("log", "wood") else 4.0,
                lambda world, _memory: self.world_risk(world) * 0.35,
                lambda world, _memory: world.item_count("log", "planks", "wood") <= 8,
            ),
            UtilityAction(
                "craft_needed_item",
                "craft_planks",
                "crafting",
                lambda world, _memory: 7.5 if world.item_count("log", "wood") > 0 and world.item_count("planks") <= 0 else 0.0,
                lambda _world, _memory: 0.5,
                lambda world, _memory: self.world_risk(world) * 0.2,
                lambda world, _memory: world.item_count("log", "wood") > 0 and world.item_count("planks") <= 0,
            ),
            UtilityAction(
                "explore_frontier",
                "sprint_wander",
                "exploration",
                lambda world, memory: 1.2 + memory.novelty_score(world.player_block) * 4.0 + (0.6 if world.is_day else 0.0),
                lambda world, _memory: 1.0 + (1.0 if world.food <= 10 else 0.0),
                lambda world, _memory: self.world_risk(world) * (0.5 if world.is_day else 0.8),
                lambda world, _memory: world.danger_level not in {"critical", "high"},
            ),
            UtilityAction(
                "scan_when_uncertain",
                "visual_scan",
                "perception",
                lambda world, memory: 2.0 + memory.novelty_score(world.player_block),
                lambda _world, _memory: 0.25,
                lambda _world, _memory: 0.0,
            ),
        ]

    def unstuck_decisions(self, world: WorldState, memory: WorldMemory) -> List[UtilityDecision]:
        return [
            UtilityDecision(
                skill="jump_forward",
                action="unstuck_jump",
                category="recovery",
                score=3.2 - memory.failure_penalty("jump_forward"),
                reward=3.4,
                cost=0.2,
                risk=self.world_risk(world) * 0.2,
                reason="position_history_low_progress",
            ),
            UtilityDecision(
                skill="dig_toward_surface",
                action="unstuck_dig",
                category="recovery",
                score=2.7 - memory.failure_penalty("dig_toward_surface"),
                reward=3.0,
                cost=0.3,
                risk=self.world_risk(world) * 0.35,
                reason="position_history_low_progress",
            ),
        ]

    def combat_reward(self, world: WorldState) -> float:
        decision = self.combat.decide(world)
        if decision.strategy == "none":
            return 0.0
        danger_bonus = {
            "critical": 12.0,
            "high": 8.0,
            "moderate": 3.0,
            "low": 1.0,
        }.get(world.danger_level, 0.0)
        if decision.strategy == "flee":
            return 12.0 + danger_bonus
        if decision.strategy == "kite":
            return 8.0 + danger_bonus
        if decision.strategy == "engage":
            return 5.0 + max(0.0, 3.0 - decision.threat_score) + danger_bonus
        return 3.0 + danger_bonus

    def world_risk(self, world: WorldState) -> float:
        risk = 0.0
        danger = world.danger_level
        if danger == "critical":
            risk += 8.0
        elif danger == "high":
            risk += 5.0
        elif danger == "moderate":
            risk += 2.5
        elif danger == "low":
            risk += 1.0
        risk += min(4.0, len(world.threats) * 0.6)
        risk += max(0.0, 10.0 - world.health) * 0.4
        if world.starving:
            risk += 1.5
        return risk

    def goal_alignment(self, world: WorldState, skill: str) -> float:
        score = 0.0
        if "diamond" in self.goal and any(term in skill for term in ("diamond", "branch", "descend", "pickaxe")):
            score += 3.0
        if any(term in self.goal for term in ("progress", "advancement", "achievement")) and any(
            term in skill for term in ("craft", "mine", "advancement", "resource")
        ):
            score += 2.0
        if ("wood" in self.goal or "tree" in self.goal) and ("log" in skill or "tree" in skill):
            score += 3.0
        if world.hungry and skill == "eat_food":
            score += 4.0
        return score

    def goal_resource_bonus(self, world: WorldState) -> float:
        if "diamond" in self.goal and world.nearest_resource("diamond"):
            return 8.0
        if any(term in self.goal for term in ("progress", "advancement", "achievement")) and world.nearest_resource("iron", "coal", "diamond"):
            return 3.0
        return 0.0

    @staticmethod
    def normalize_goal(goal: str) -> str:
        return " ".join(str(goal or "").strip().lower().replace("_", " ").split())
