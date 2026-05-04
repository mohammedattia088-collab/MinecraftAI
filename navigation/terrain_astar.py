from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from ..perception.world_state import Vec3, WorldState, manhattan
except ImportError:
    from perception.world_state import Vec3, WorldState, manhattan


@dataclass(order=True)
class PathNode:
    priority: float
    pos: Vec3 = field(compare=False)
    cost: float = field(default=0.0, compare=False)
    parent: Optional[Vec3] = field(default=None, compare=False)


@dataclass
class PathPlan:
    path: List[Vec3]
    cost: float
    explored: int
    reached: bool
    reason: str = "ok"
    escape_score: float = 0.0

    @property
    def next_waypoint(self) -> Optional[Vec3]:
        if len(self.path) >= 2:
            return self.path[1]
        if self.path:
            return self.path[0]
        return None


class TerrainAStar:
    """
    Local terrain-aware A* over bridge observations.

    The bridge sends sparse nearby block samples, so unknown cells are allowed
    with a configurable penalty. Known hazards and body-blocking cells are avoided.
    """

    CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
    DIAGONALS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))

    def __init__(
        self,
        world: WorldState,
        *,
        max_jump: int = 1,
        max_safe_fall: int = 3,
        hunger_weight: float = 0.035,
        danger_weight: float = 1.0,
        unknown_cost: float = 1.65,
    ) -> None:
        self.world = world
        self.max_jump = max_jump
        self.max_safe_fall = max_safe_fall
        self.hunger_weight = hunger_weight
        self.danger_weight = danger_weight
        self.unknown_cost = unknown_cost

    def plan(
        self,
        start: Vec3,
        goal: Vec3,
        *,
        max_nodes: int = 4096,
        allow_partial: bool = True,
    ) -> PathPlan:
        if self.is_body_blocked(start):
            start = self.nearest_known_open(start) or start
        goals = self.goal_candidates(goal)
        if not goals:
            return PathPlan([], math.inf, 0, False, "no_reachable_goal")

        frontier: List[PathNode] = []
        heapq.heappush(frontier, PathNode(0.0, start, 0.0, None))
        came_from: Dict[Vec3, Optional[Vec3]] = {start: None}
        cost_so_far: Dict[Vec3, float] = {start: 0.0}
        best = start
        best_h = min(manhattan(start, candidate) for candidate in goals)
        explored = 0

        while frontier and explored < max_nodes:
            current = heapq.heappop(frontier)
            explored += 1
            if current.pos in goals:
                return PathPlan(self.reconstruct(came_from, current.pos), current.cost, explored, True)

            h = min(manhattan(current.pos, candidate) for candidate in goals)
            if h < best_h:
                best = current.pos
                best_h = h

            for neighbor, move_cost in self.neighbors(current.pos):
                new_cost = cost_so_far[current.pos] + move_cost
                if new_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current.pos
                heuristic = min(self.heuristic(neighbor, candidate) for candidate in goals)
                heapq.heappush(frontier, PathNode(new_cost + heuristic, neighbor, new_cost, current.pos))

        if allow_partial and best in came_from:
            return PathPlan(
                self.reconstruct(came_from, best),
                cost_so_far.get(best, math.inf),
                explored,
                False,
                "partial" if explored < max_nodes else "node_budget_exhausted",
            )
        return PathPlan([], math.inf, explored, False, "node_budget_exhausted")

    def escape_paths(
        self,
        *,
        radius: int = 8,
        samples: int = 16,
        max_nodes: int = 2048,
    ) -> List[PathPlan]:
        start = self.world.player_block
        threat_x = 0.0
        threat_z = 0.0
        for threat in self.world.threats:
            weight = max(0.1, threat.score) / max(1.0, threat.distance)
            threat_x += (self.world.player_pos[0] - threat.pos[0]) * weight
            threat_z += (self.world.player_pos[2] - threat.pos[2]) * weight

        if abs(threat_x) + abs(threat_z) < 0.001:
            yaw = math.radians(self.world.yaw)
            threat_x = -math.sin(yaw)
            threat_z = math.cos(yaw)

        base_angle = math.atan2(threat_z, threat_x)
        candidates: List[Tuple[float, Vec3]] = []
        for index in range(samples):
            offset = (index - samples // 2) * (math.pi / max(6, samples))
            angle = base_angle + offset
            x = start[0] + int(round(math.cos(angle) * radius))
            z = start[2] + int(round(math.sin(angle) * radius))
            candidate = (x, start[1], z)
            direct_danger = self.threat_penalty(candidate) + self.world.terrain.danger_at(candidate)
            candidates.append((direct_danger, candidate))

        plans: List[PathPlan] = []
        for _, candidate in sorted(candidates, key=lambda item: item[0])[: max(4, samples // 2)]:
            plan = self.plan(start, candidate, max_nodes=max_nodes, allow_partial=True)
            if not plan.path:
                continue
            end = plan.path[-1]
            plan.escape_score = self.escape_score(end, plan.cost)
            plans.append(plan)
        plans.sort(key=lambda plan: plan.escape_score, reverse=True)
        return plans

    def neighbors(self, pos: Vec3) -> Iterable[Tuple[Vec3, float]]:
        directions = self.CARDINALS + self.DIAGONALS
        for dx, dz in directions:
            diagonal = dx != 0 and dz != 0
            for dy in (0, 1, -1, -2, -3):
                candidate = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                feasible, penalty = self.feasible_step(pos, candidate, diagonal)
                if feasible:
                    yield candidate, penalty
                    break

    def feasible_step(self, current: Vec3, candidate: Vec3, diagonal: bool) -> Tuple[bool, float]:
        dy = candidate[1] - current[1]
        if dy > self.max_jump:
            return False, math.inf
        if dy < -self.max_safe_fall:
            return False, math.inf
        if self.is_body_blocked(candidate):
            return False, math.inf
        if diagonal:
            side_a = (candidate[0], candidate[1], current[2])
            side_b = (current[0], candidate[1], candidate[2])
            if self.is_body_blocked(side_a) or self.is_body_blocked(side_b):
                return False, math.inf

        ground = (candidate[0], candidate[1] - 1, candidate[2])
        ground_block = self.world.blocks.get(ground)
        unknown_ground = ground_block is None
        if ground_block is not None and not ground_block.obstacle:
            if dy >= 0:
                return False, math.inf

        distance_cost = math.sqrt(2.0) if diagonal else 1.0
        jump_penalty = max(0, dy) * 2.2
        fall_penalty = self.fall_damage_penalty(-dy)
        hunger_penalty = (1.0 if self.world.food <= 14 else 0.35) * self.hunger_weight * distance_cost
        danger_penalty = self.threat_penalty(candidate) * self.danger_weight
        hazard_penalty = self.world.terrain.danger_at(candidate)
        unknown_penalty = self.unknown_cost if unknown_ground else 0.0
        terrain_cost = self.world.terrain.cost_at(candidate)
        if math.isinf(terrain_cost):
            return False, math.inf
        return True, distance_cost + jump_penalty + fall_penalty + hunger_penalty + danger_penalty + hazard_penalty + unknown_penalty

    def is_body_blocked(self, feet: Vec3) -> bool:
        head = (feet[0], feet[1] + 1, feet[2])
        for pos in (feet, head):
            block = self.world.blocks.get(pos)
            if block is not None and block.obstacle:
                return True
            if pos in self.world.terrain.hazards:
                return True
        return False

    def nearest_known_open(self, pos: Vec3, radius: int = 3) -> Optional[Vec3]:
        best: Optional[Vec3] = None
        best_distance = math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-1, 2):
                for dz in range(-radius, radius + 1):
                    candidate = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    if self.is_body_blocked(candidate):
                        continue
                    distance = manhattan(pos, candidate)
                    if distance < best_distance:
                        best = candidate
                        best_distance = distance
        return best

    def goal_candidates(self, goal: Vec3) -> Set[Vec3]:
        candidates: Set[Vec3] = set()
        search = [goal]
        for dx, dz in self.CARDINALS + self.DIAGONALS:
            search.append((goal[0] + dx, goal[1], goal[2] + dz))
            search.append((goal[0] + dx, goal[1] - 1, goal[2] + dz))
            search.append((goal[0] + dx, goal[1] + 1, goal[2] + dz))
        for candidate in search:
            if not self.is_body_blocked(candidate):
                candidates.add(candidate)
        return candidates

    def fall_damage_penalty(self, fall_distance: int) -> float:
        if fall_distance <= self.max_safe_fall:
            return max(0, fall_distance) * 0.25
        return 25.0 + (fall_distance - self.max_safe_fall) * 8.0

    def threat_penalty(self, pos: Vec3) -> float:
        px, py, pz = pos
        penalty = 0.0
        for threat in self.world.threats:
            dx = px + 0.5 - threat.pos[0]
            dy = py + 0.5 - threat.pos[1]
            dz = pz + 0.5 - threat.pos[2]
            distance = max(0.5, math.sqrt(dx * dx + dy * dy + dz * dz))
            influence = max(0.0, 10.0 - distance)
            penalty += influence * max(1.0, threat.score) * 0.08
        return penalty

    def escape_score(self, pos: Vec3, path_cost: float) -> float:
        nearest_threat = min(
            (
                math.sqrt((pos[0] + 0.5 - threat.pos[0]) ** 2 + (pos[2] + 0.5 - threat.pos[2]) ** 2)
                for threat in self.world.threats
            ),
            default=16.0,
        )
        light_bonus = 0.2 if self.world.is_day else 0.0
        return nearest_threat * 1.4 + light_bonus - path_cost * 0.18 - self.world.terrain.danger_at(pos)

    @staticmethod
    def heuristic(a: Vec3, b: Vec3) -> float:
        return abs(a[0] - b[0]) + abs(a[2] - b[2]) + abs(a[1] - b[1]) * 1.8

    @staticmethod
    def reconstruct(came_from: Dict[Vec3, Optional[Vec3]], current: Vec3) -> List[Vec3]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path
