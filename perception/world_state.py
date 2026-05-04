from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple


Vec3 = Tuple[int, int, int]
FloatVec3 = Tuple[float, float, float]

HOSTILE_TERMS = (
    "zombie", "skeleton", "creeper", "spider", "enderman", "witch", "slime",
    "drowned", "husk", "stray", "phantom", "pillager", "vindicator", "ravager",
    "warden", "blaze", "ghast", "guardian", "shulker", "silverfish", "magma_cube",
    "hoglin", "zoglin", "piglin_brute", "projectile",
)

RESOURCE_TERMS = (
    "log", "wood", "ore", "coal", "iron", "copper", "gold", "diamond",
    "redstone", "lapis", "emerald", "ancient_debris", "cobblestone", "deepslate",
)

VALUABLE_ITEM_TERMS = (
    "apple", "bread", "beef", "porkchop", "chicken", "mutton", "cod", "salmon",
    "carrot", "potato", "melon", "log", "planks", "stick", "coal", "charcoal",
    "cobblestone", "deepslate", "iron", "gold", "diamond", "pickaxe", "axe",
    "sword", "shovel", "crafting_table", "furnace", "torch", "sapling", "wheat",
    "seed",
)

LOW_VALUE_ITEM_TERMS = (
    "dirt", "grass_block", "coarse_dirt", "podzol", "snowball", "rotten_flesh",
    "sand", "gravel", "flint",
)

FOOD_ENTITY_TERMS = (
    "cow", "pig", "chicken", "sheep", "rabbit", "cod", "salmon",
)

DANGEROUS_BLOCK_TERMS = (
    "lava", "fire", "magma_block", "cactus", "powder_snow", "sweet_berry_bush",
    "campfire", "soul_fire",
)

FLUID_TERMS = ("water", "lava")

PASSABLE_BLOCK_TERMS = (
    "air", "cave_air", "void_air", "grass", "fern", "flower", "sapling",
    "torch", "vine", "snow", "seagrass", "kelp",
)

UNBREAKABLE_BLOCK_TERMS = (
    "bedrock", "barrier", "command_block", "end_portal", "end_gateway",
    "jigsaw", "structure_block",
)


@dataclass(frozen=True)
class BlockInfo:
    pos: Vec3
    block: str
    distance: float = 0.0
    hardness: float = 0.0
    walkable: bool = False
    obstacle: bool = True
    dangerous: bool = False
    resource: bool = False
    fluid: bool = False
    unbreakable: bool = False
    light: int = 0
    can_see_sky: bool = False
    cost: float = 1.0


@dataclass(frozen=True)
class EntityInfo:
    entity_id: int
    entity_type: str
    pos: FloatVec3
    distance: float = 0.0
    hostile: bool = False
    projectile: bool = False
    item: bool = False
    food_source: bool = False
    line_of_sight: bool = False
    health: float = 0.0
    max_health: float = 0.0
    threat_score: float = 0.0
    item_name: str = ""
    item_count: int = 0
    moving_toward_player: bool = False


@dataclass(frozen=True)
class ResourceInfo:
    pos: Vec3
    resource_type: str
    distance: float
    source: str = "block"
    score: float = 0.0


@dataclass(frozen=True)
class ThreatInfo:
    entity: Optional[EntityInfo]
    pos: FloatVec3
    score: float
    distance: float
    immediate: bool = False
    kind: str = "entity"


@dataclass
class TerrainCostMap:
    costs: Dict[Vec3, float] = field(default_factory=dict)
    blocked: Set[Vec3] = field(default_factory=set)
    hazards: Set[Vec3] = field(default_factory=set)
    resources: Set[Vec3] = field(default_factory=set)
    unknown_cost: float = 1.65

    def cost_at(self, pos: Vec3) -> float:
        if pos in self.blocked:
            return math.inf
        return float(self.costs.get(pos, self.unknown_cost))

    def danger_at(self, pos: Vec3, radius: int = 2) -> float:
        px, py, pz = pos
        danger = 0.0
        for hx, hy, hz in self.hazards:
            d = abs(px - hx) + abs(py - hy) + abs(pz - hz)
            if d <= radius:
                danger += (radius + 1 - d) * 4.0
        return danger


@dataclass
class WorldState:
    raw: Dict[str, Any]
    player_pos: FloatVec3
    player_block: Vec3
    health: float
    max_health: float
    food: float
    saturation: float
    yaw: float
    pitch: float
    is_day: bool
    danger_level: str
    blocks: Dict[Vec3, BlockInfo] = field(default_factory=dict)
    entities: List[EntityInfo] = field(default_factory=list)
    threats: List[ThreatInfo] = field(default_factory=list)
    resources: List[ResourceInfo] = field(default_factory=list)
    terrain: TerrainCostMap = field(default_factory=TerrainCostMap)
    timestamp: float = field(default_factory=time.time)

    @property
    def alive(self) -> bool:
        return self.health > 0.0 and not bool(self.raw.get("player", {}).get("is_dead"))

    @property
    def hungry(self) -> bool:
        return self.food <= 14.0 or (self.health <= max(1.0, self.max_health) * 0.55 and self.food < 20.0)

    @property
    def starving(self) -> bool:
        return self.food <= 7.0

    @property
    def in_environmental_danger(self) -> bool:
        player = self.raw.get("player", {})
        return (
            bool(player.get("is_in_lava"))
            or bool(player.get("is_on_fire"))
            or float(player.get("fall_distance", 0.0)) > 7.0
            or any(threat.kind == "hazard" and threat.distance <= 2.5 for threat in self.threats)
        )

    def nearest_resource(self, *terms: str) -> Optional[ResourceInfo]:
        normalized = tuple(term.lower() for term in terms if term)
        candidates = self.resources
        if normalized:
            candidates = [
                resource for resource in candidates
                if any(term in resource.resource_type.lower() for term in normalized)
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda resource: resource.score)

    def hostile_count(self, radius: float = 12.0) -> int:
        return sum(1 for entity in self.entities if entity.hostile and entity.distance <= radius)

    def item_count(self, *terms: str) -> int:
        normalized = tuple(term.lower() for term in terms)
        total = 0
        for item in self.raw.get("inventory", {}).get("items", []):
            item_id = str(item.get("item", "")).lower()
            if not normalized or any(term in item_id for term in normalized):
                total += int(item.get("count", 0))
        return total

    def edible_count(self) -> int:
        return sum(
            int(item.get("count", 0))
            for item in self.raw.get("inventory", {}).get("items", [])
            if bool(item.get("is_edible"))
        )


class WorldMemory:
    """Small rolling map so exploration has memory instead of pure wandering."""

    def __init__(self, max_positions: int = 2048) -> None:
        self.visited_positions: Deque[Vec3] = deque(maxlen=max_positions)
        self.visited_chunks: Set[Tuple[int, int]] = set()
        self.seen_resources: Dict[Vec3, ResourceInfo] = {}
        self.failures_by_skill: Dict[str, int] = defaultdict(int)
        self.last_seen_at: Dict[Vec3, float] = {}
        self.skill_aliases: Dict[str, Tuple[str, ...]] = {
            "collect_visible_item": ("collect_item",),
            "collect_item": ("collect_visible_item",),
        }

    def update(self, world: WorldState) -> None:
        self.visited_positions.append(world.player_block)
        self.visited_chunks.add((world.player_block[0] // 16, world.player_block[2] // 16))
        now = time.time()
        for resource in world.resources:
            self.seen_resources[resource.pos] = resource
            self.last_seen_at[resource.pos] = now

    def novelty_score(self, pos: Optional[Vec3] = None) -> float:
        pos = pos or (self.visited_positions[-1] if self.visited_positions else (0, 0, 0))
        if not self.visited_positions:
            return 1.0
        recent = list(self.visited_positions)[-96:]
        nearest = min(manhattan(pos, sample) for sample in recent)
        return max(0.0, min(1.0, nearest / 24.0))

    def record_result(self, skill: str, success: bool) -> None:
        related = (skill, *self.skill_aliases.get(skill, ()))
        if success:
            for name in related:
                self.failures_by_skill.pop(name, None)
        else:
            for name in related:
                self.failures_by_skill[name] += 1

    def failure_penalty(self, skill: str) -> float:
        return min(8.0, self.failures_by_skill.get(skill, 0) * 1.25)


def build_world_state(observation: Dict[str, Any]) -> WorldState:
    player = observation.get("player", {})
    position = player.get("position", {})
    player_pos = (
        float(position.get("x", 0.0)),
        float(position.get("y", 0.0)),
        float(position.get("z", 0.0)),
    )
    player_block = (
        int(math.floor(player_pos[0])),
        int(math.floor(player_pos[1])),
        int(math.floor(player_pos[2])),
    )
    environment = observation.get("environment", {})
    threats_payload = observation.get("threats", {})

    blocks: Dict[Vec3, BlockInfo] = {}
    terrain = TerrainCostMap()
    resources: List[ResourceInfo] = []
    for raw_block in observation.get("nearby_blocks", observation.get("blocks", [])) or []:
        block = parse_block(raw_block)
        blocks[block.pos] = block
        if block.obstacle:
            terrain.blocked.add(block.pos)
        if block.dangerous:
            terrain.hazards.add(block.pos)
        if block.resource:
            terrain.resources.add(block.pos)
            resources.append(
                ResourceInfo(
                    pos=block.pos,
                    resource_type=block.block,
                    distance=block.distance,
                    source="block",
                    score=score_resource(block),
                )
            )
        terrain.costs[block.pos] = block.cost

    entities = [parse_entity(raw_entity) for raw_entity in observation.get("nearby_entities", []) or []]
    for entity in entities:
        if entity.item:
            resources.append(
                ResourceInfo(
                    pos=(int(round(entity.pos[0])), int(round(entity.pos[1])), int(round(entity.pos[2]))),
                    resource_type=entity.item_name or entity.entity_type,
                    distance=entity.distance,
                    source="item",
                    score=score_item_resource(entity),
                )
            )

    threat_list: List[ThreatInfo] = []
    for entity in entities:
        if entity.hostile or entity.projectile or entity.threat_score > 0.0:
            threat_list.append(
                ThreatInfo(
                    entity=entity,
                    pos=entity.pos,
                    score=max(entity.threat_score, 1.0),
                    distance=entity.distance,
                    immediate=entity.distance <= 4.0 or entity.threat_score >= 12.0,
                    kind="entity",
                )
            )

    for raw_threat in threats_payload.get("entities", []) or []:
        entity = parse_entity(raw_threat)
        if not any(existing.entity and existing.entity.entity_id == entity.entity_id for existing in threat_list):
            threat_list.append(
                ThreatInfo(
                    entity=entity,
                    pos=entity.pos,
                    score=max(entity.threat_score, float(raw_threat.get("threat_score", 1.0))),
                    distance=entity.distance,
                    immediate=bool(raw_threat.get("immediate", entity.distance <= 4.0)),
                    kind="entity",
                )
            )

    for raw_hazard in threats_payload.get("hazards", []) or observation.get("survival", {}).get("nearby_hazards", []) or []:
        hazard = parse_block(raw_hazard)
        threat_list.append(
            ThreatInfo(
                entity=None,
                pos=(float(hazard.pos[0]) + 0.5, float(hazard.pos[1]) + 0.5, float(hazard.pos[2]) + 0.5),
                score=max(3.0, 10.0 - hazard.distance),
                distance=hazard.distance,
                immediate=hazard.distance <= 2.5,
                kind="hazard",
            )
        )
        terrain.hazards.add(hazard.pos)

    threat_list.sort(key=lambda threat: threat.score, reverse=True)
    resources.sort(key=lambda resource: resource.score, reverse=True)

    return WorldState(
        raw=observation,
        player_pos=player_pos,
        player_block=player_block,
        health=float(player.get("health", 20.0)),
        max_health=float(player.get("max_health", 20.0)),
        food=float(player.get("food", 20.0)),
        saturation=float(player.get("saturation", 0.0)),
        yaw=float(player.get("yaw", 0.0)),
        pitch=float(player.get("pitch", 0.0)),
        is_day=bool(environment.get("is_day", True)),
        danger_level=str(threats_payload.get("danger_level", "none")),
        blocks=blocks,
        entities=entities,
        threats=threat_list,
        resources=resources,
        terrain=terrain,
    )


def parse_block(raw: Dict[str, Any]) -> BlockInfo:
    block_id = str(raw.get("block", "minecraft:air")).lower()
    pos = (
        int(raw.get("x", 0)),
        int(raw.get("y", 0)),
        int(raw.get("z", 0)),
    )
    dangerous = bool(raw.get("hazard")) or any(term in block_id for term in DANGEROUS_BLOCK_TERMS)
    resource = bool(raw.get("resource")) or any(term in block_id for term in RESOURCE_TERMS)
    fluid = bool(raw.get("is_fluid")) or any(term in block_id for term in FLUID_TERMS)
    passable = any(term in block_id for term in PASSABLE_BLOCK_TERMS)
    unbreakable = any(term in block_id for term in UNBREAKABLE_BLOCK_TERMS)
    hardness = float(raw.get("hardness", 0.0))
    obstacle = not passable and not block_id.endswith(":air")
    cost = 1.0
    if obstacle:
        cost += 6.0
    if dangerous:
        cost += 100.0
    if fluid:
        cost += 6.0
    if resource:
        cost += 0.75
    if hardness < 0.0 or unbreakable:
        cost = math.inf
    return BlockInfo(
        pos=pos,
        block=block_id,
        distance=float(raw.get("distance", 0.0)),
        hardness=hardness,
        walkable=not obstacle and not dangerous,
        obstacle=obstacle,
        dangerous=dangerous,
        resource=resource,
        fluid=fluid,
        unbreakable=unbreakable,
        light=int(raw.get("block_light", raw.get("combined_light", 0))),
        can_see_sky=bool(raw.get("can_see_sky", False)),
        cost=cost,
    )


def parse_entity(raw: Dict[str, Any]) -> EntityInfo:
    entity_type = str(raw.get("type", "")).lower()
    item_name = str(raw.get("item_name", "")).lower()
    hostile = bool(raw.get("hostile")) or bool(raw.get("threat")) or any(term in entity_type for term in HOSTILE_TERMS)
    projectile = bool(raw.get("projectile")) or "arrow" in entity_type or "fireball" in entity_type
    item = bool(raw.get("item")) or entity_type == "minecraft:item"
    food_source = any(term in entity_type for term in FOOD_ENTITY_TERMS)
    return EntityInfo(
        entity_id=int(raw.get("id", raw.get("entity_id", -1))),
        entity_type=entity_type,
        pos=(float(raw.get("x", 0.0)), float(raw.get("y", 0.0)), float(raw.get("z", 0.0))),
        distance=float(raw.get("distance", 999.0)),
        hostile=hostile,
        projectile=projectile,
        item=item,
        food_source=food_source,
        line_of_sight=bool(raw.get("line_of_sight", raw.get("visible", False))),
        health=float(raw.get("health", 0.0)),
        max_health=float(raw.get("max_health", 0.0)),
        threat_score=float(raw.get("threat_score", raw.get("combat_score", 0.0))),
        item_name=item_name,
        item_count=int(raw.get("item_count", 0)),
        moving_toward_player=bool(raw.get("moving_toward_player", False)),
    )


def score_resource(block: BlockInfo) -> float:
    value = 1.0
    name = block.block
    for term, bonus in (
        ("diamond", 12.0),
        ("ancient_debris", 15.0),
        ("iron", 6.0),
        ("coal", 4.0),
        ("log", 5.0),
        ("cobblestone", 3.0),
        ("deepslate", 2.5),
    ):
        if term in name:
            value += bonus
            break
    distance_penalty = min(8.0, block.distance * 0.18)
    hazard_penalty = 4.0 if block.dangerous else 0.0
    return value - distance_penalty - hazard_penalty


def score_item_resource(entity: EntityInfo) -> float:
    name = (entity.item_name or entity.entity_type).replace("minecraft:", "").lower()
    if any(term in name for term in LOW_VALUE_ITEM_TERMS):
        value = 0.45
    elif any(term in name for term in VALUABLE_ITEM_TERMS):
        value = 5.0
    else:
        value = 1.25

    if any(term in name for term in ("apple", "bread", "beef", "porkchop", "chicken", "mutton", "cod", "salmon", "carrot", "potato")):
        value += 2.5
    if any(term in name for term in ("iron", "diamond", "gold", "pickaxe", "sword")):
        value += 2.0

    count_bonus = min(2.0, max(0, entity.item_count) * 0.12)
    distance_penalty = min(5.0, entity.distance * 0.35)
    return max(0.05, value + count_bonus - distance_penalty)


def manhattan(a: Vec3, b: Vec3) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
