from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..bridge import BridgeClient, BridgeError
    from ..mechanics import BasicMechanics
    from ..perception.world_state import WorldMemory, build_world_state
    from ..planning.utility import UtilityPlanner
except ImportError:
    from bridge import BridgeClient, BridgeError
    from mechanics import BasicMechanics
    from perception.world_state import WorldMemory, build_world_state
    from planning.utility import UtilityPlanner


@dataclass
class SeedRunMetrics:
    seed: str
    world: str
    steps: int = 0
    survived: bool = True
    deaths: int = 0
    time_to_wood: Optional[float] = None
    time_to_pickaxe: Optional[float] = None
    time_to_stone: Optional[float] = None
    final_health: float = 0.0
    final_food: float = 0.0
    final_stage: int = 0
    failures: List[str] = field(default_factory=list)
    actions: Dict[str, int] = field(default_factory=dict)


@dataclass
class MultiSeedResult:
    runs: List[SeedRunMetrics]

    def summary(self) -> Dict[str, Any]:
        total = max(1, len(self.runs))
        survived = sum(1 for run in self.runs if run.survived)
        return {
            "run_count": len(self.runs),
            "survival_rate": survived / total,
            "average_final_stage": sum(run.final_stage for run in self.runs) / total,
            "average_time_to_wood": average_present(run.time_to_wood for run in self.runs),
            "average_time_to_pickaxe": average_present(run.time_to_pickaxe for run in self.runs),
            "failures": [failure for run in self.runs for failure in run.failures],
        }

    def as_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "runs": [asdict(run) for run in self.runs]}


class MultiSeedValidator:
    """
    Automated evaluator for survival/progression across world labels.

    The bridge can open existing singleplayer worlds. If a seed world does not
    exist yet, create it in Minecraft once with the matching world name, then run
    this harness repeatedly for regression testing.
    """

    def __init__(
        self,
        client: BridgeClient,
        *,
        world_prefix: str = "AISeed",
        steps: int = 120,
        step_delay: float = 0.15,
        goal: str = "survive and progress",
    ) -> None:
        self.client = client
        self.mechanics = BasicMechanics(client, step_delay=step_delay)
        self.planner = UtilityPlanner(goal)
        self.memory = WorldMemory(max_positions=1024)
        self.world_prefix = world_prefix
        self.steps = steps
        self.step_delay = step_delay

    async def run(self, seeds: Iterable[str]) -> MultiSeedResult:
        await self.client.ensure_connected()
        runs: List[SeedRunMetrics] = []
        for seed in seeds:
            runs.append(await self.run_seed(str(seed)))
        return MultiSeedResult(runs)

    async def run_seed(self, seed: str) -> SeedRunMetrics:
        world_name = f"{self.world_prefix}_{seed}"
        metrics = SeedRunMetrics(seed=seed, world=world_name)
        self.memory = WorldMemory(max_positions=1024)
        started = time.monotonic()

        try:
            await self.client.request("open_singleplayer_world", world=world_name)
            await asyncio.sleep(4.0)
        except (BridgeError, OSError, asyncio.TimeoutError) as exc:
            metrics.failures.append(f"world_open:{exc}")

        for step in range(self.steps):
            observation = await self.safe_observe(metrics)
            if observation is None:
                break
            world = build_world_state(observation)
            metrics.steps = step + 1
            metrics.final_health = world.health
            metrics.final_food = world.food
            metrics.final_stage = self.stage_score(world.raw)

            if not world.alive:
                metrics.deaths += 1
                metrics.survived = False
                try:
                    await self.client.request("respawn")
                except BridgeError as exc:
                    metrics.failures.append(f"respawn:{exc}")
                    break

            self.record_milestones(metrics, world.raw, started)
            decision = self.planner.decide(world, self.memory)
            self.memory.update(world)
            metrics.actions[decision.skill] = metrics.actions.get(decision.skill, 0) + 1
            result = await self.mechanics.run(decision.skill, observation)
            self.memory.record_result(result.name, result.success)
            if not result.success:
                metrics.failures.append(f"{decision.skill}:{result.details.get('error', result.details.get('reason', 'failed'))}")
            await asyncio.sleep(self.step_delay)

        await self.mechanics.stop_all()
        return metrics

    async def safe_observe(self, metrics: SeedRunMetrics) -> Optional[Dict[str, Any]]:
        try:
            return await self.mechanics.observe()
        except (BridgeError, OSError, asyncio.TimeoutError) as exc:
            metrics.failures.append(f"observe:{exc}")
            return None

    def record_milestones(self, metrics: SeedRunMetrics, observation: Dict[str, Any], started: float) -> None:
        elapsed = time.monotonic() - started
        inventory = observation.get("inventory", {})
        if metrics.time_to_wood is None and self.item_count(inventory, ("log", "wood", "planks")) > 0:
            metrics.time_to_wood = elapsed
        if metrics.time_to_pickaxe is None and self.item_count(inventory, ("pickaxe",)) > 0:
            metrics.time_to_pickaxe = elapsed
        if metrics.time_to_stone is None and self.item_count(inventory, ("cobblestone", "cobbled_deepslate")) > 0:
            metrics.time_to_stone = elapsed

    @staticmethod
    def stage_score(observation: Dict[str, Any]) -> int:
        inventory = observation.get("inventory", {})
        score = 0
        for terms in (
            ("log", "wood", "planks"),
            ("crafting_table",),
            ("stick",),
            ("pickaxe",),
            ("cobblestone", "cobbled_deepslate"),
            ("stone_pickaxe",),
            ("iron",),
            ("diamond",),
        ):
            if MultiSeedValidator.item_count(inventory, terms) > 0:
                score += 1
        return score

    @staticmethod
    def item_count(inventory: Dict[str, Any], terms: Iterable[str]) -> int:
        normalized = tuple(term.lower() for term in terms)
        return sum(
            int(item.get("count", 0))
            for item in inventory.get("items", [])
            if any(term in str(item.get("item", "")).lower() for term in normalized)
        )


def average_present(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


async def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Run MinecraftAI multi-seed/world validation.")
    parser.add_argument("seeds", nargs="+")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25575)
    parser.add_argument("--auth-token", default="")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--world-prefix", default="AISeed")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    client = BridgeClient(args.host, args.port, args.auth_token, timeout=8.0)
    validator = MultiSeedValidator(client, world_prefix=args.world_prefix, steps=args.steps)
    try:
        result = await validator.run(args.seeds)
    finally:
        await client.close()

    payload = result.as_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    asyncio.run(cli_main())
