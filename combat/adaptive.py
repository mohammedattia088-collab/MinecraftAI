from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from ..navigation import PathPlan, TerrainAStar
    from ..perception.world_state import EntityInfo, WorldState
except ImportError:
    from navigation import PathPlan, TerrainAStar
    from perception.world_state import EntityInfo, WorldState


@dataclass(frozen=True)
class CombatDecision:
    strategy: str
    target: Optional[EntityInfo]
    threat_score: float
    mob_count: int
    desired_distance: float
    attack_interval: float
    strafe: float
    reason: str
    escape_path: Optional[PathPlan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CombatTactics:
    """Adaptive combat layer: score threats, choose engage/kite/flee/recover."""

    def __init__(self, *, combat_radius: float = 10.0, flee_radius: float = 18.0) -> None:
        self.combat_radius = combat_radius
        self.flee_radius = flee_radius

    def decide(self, world: WorldState) -> CombatDecision:
        hostiles = [
            entity for entity in world.entities
            if entity.hostile and entity.distance <= self.flee_radius
        ]
        hostiles.sort(key=lambda entity: (entity.threat_score, -entity.distance), reverse=True)
        target = hostiles[0] if hostiles else None
        mob_count = len(hostiles)
        bridge_score = float(world.raw.get("threats", {}).get("score", 0.0))
        nearest = min((entity.distance for entity in hostiles), default=99.0)
        health = max(1.0, world.health)
        armor = float(world.raw.get("player", {}).get("armor_value", 0.0))
        weapon_score = float(world.raw.get("combat", {}).get("weapon_score", 0.0))
        threat_score = (bridge_score + mob_count * 4.0 + max(0.0, 8.0 - nearest) * 1.5) / (health + armor * 0.35)

        if world.in_environmental_danger:
            return self._decision("flee", target, threat_score, mob_count, 12.0, "environmental_danger", world)
        if not hostiles:
            return CombatDecision("none", None, 0.0, 0, 0.0, 0.0, 0.0, "no_hostiles")
        if world.health <= 6.0 or threat_score >= 2.8 or mob_count >= 5:
            return self._decision("flee", target, threat_score, mob_count, 14.0, "overwhelmed", world)
        if mob_count >= 3 or (nearest <= 3.5 and world.health <= 12.0):
            return self._decision("kite", target, threat_score, mob_count, 7.0, "multiple_or_close", world)
        if weapon_score <= 0.0 and nearest <= 5.0:
            return self._decision("kite", target, threat_score, mob_count, 7.5, "no_weapon", world)
        if nearest <= self.combat_radius:
            return self._decision("engage", target, threat_score, mob_count, 3.1, "favorable_duel", world)
        return self._decision("track", target, threat_score, mob_count, 6.0, "target_far", world)

    def _decision(
        self,
        strategy: str,
        target: Optional[EntityInfo],
        threat_score: float,
        mob_count: int,
        desired_distance: float,
        reason: str,
        world: WorldState,
    ) -> CombatDecision:
        attack_interval = self.variable_attack_interval(world, target)
        strafe = random.choice((-1.0, -0.75, 0.75, 1.0))
        escape_path = None
        if strategy in {"flee", "kite"}:
            escape_candidates = TerrainAStar(world).escape_paths(radius=8 if strategy == "kite" else 12, samples=14)
            escape_path = escape_candidates[0] if escape_candidates else None
        return CombatDecision(
            strategy=strategy,
            target=target,
            threat_score=threat_score,
            mob_count=mob_count,
            desired_distance=desired_distance,
            attack_interval=attack_interval,
            strafe=strafe,
            reason=reason,
            escape_path=escape_path,
            metadata={
                "health": world.health,
                "food": world.food,
                "danger_level": world.danger_level,
                "weapon_score": world.raw.get("combat", {}).get("weapon_score"),
            },
        )

    @staticmethod
    def variable_attack_interval(world: WorldState, target: Optional[EntityInfo]) -> float:
        attack_strength = float(world.raw.get("combat", {}).get("attack_strength", 1.0))
        base = 0.18 if attack_strength >= 0.9 else 0.32
        if target is not None:
            base += min(0.22, max(0.0, target.distance - 3.0) * 0.035)
        fatigue = 0.10 if world.food <= 8 else 0.0
        return max(0.12, min(0.8, base + fatigue + random.uniform(-0.045, 0.075)))

    @staticmethod
    def threat_vector_yaw(world: WorldState) -> float:
        away_x = 0.0
        away_z = 0.0
        for threat in world.threats:
            distance = max(1.0, threat.distance)
            away_x += (world.player_pos[0] - threat.pos[0]) * threat.score / (distance * distance)
            away_z += (world.player_pos[2] - threat.pos[2]) * threat.score / (distance * distance)
        if abs(away_x) + abs(away_z) < 0.0001:
            return world.yaw + 180.0
        return math.degrees(math.atan2(-away_x, away_z))

