import argparse
import asyncio
import time
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Iterable, Tuple

try:
    from .bridge import BridgeClient, BridgeError
    from .mechanics import BasicMechanics, SkillResult
except ImportError:
    from bridge import BridgeClient, BridgeError
    from mechanics import BasicMechanics, SkillResult


SmokeCall = Callable[[], Awaitable[Dict[str, Any]]]


def summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"status": payload.get("status", "success")}
    for key in (
        "action_id",
        "version",
        "minecraft_version",
        "bridge_api_level",
        "generic_crafting_engine",
        "vanilla_crafting_recipes",
        "generic_supported_count",
        "current_inventory_craftable",
        "returned_count",
        "crafted",
        "reason",
        "opened",
        "available",
        "completed_count",
        "missing_count",
        "queued",
        "world",
        "phase",
        "found",
        "block",
    ):
        if key in payload:
            summary[key] = payload[key]
    for key in ("x", "y", "z"):
        if key in payload:
            summary[key] = payload[key]

    if "player" in payload:
        player = payload["player"]
        summary["player"] = {
            "health": player.get("health"),
            "food": player.get("food"),
            "position": player.get("position"),
        }
    if "visual_summary" in payload:
        visual = payload["visual_summary"]
        summary["visual_summary"] = {
            "recommended_focus": visual.get("recommended_focus"),
            "resource_count": visual.get("resource_count"),
            "hazard_count": visual.get("hazard_count"),
            "opening_count": visual.get("opening_count"),
        }
    if "crafting_context" in payload:
        summary["crafting_context"] = payload["crafting_context"]
    if "recipe" in payload and isinstance(payload["recipe"], dict):
        recipe = payload["recipe"]
        summary["recipe"] = {
            "id": recipe.get("id"),
            "result": recipe.get("result"),
            "shape": recipe.get("shape"),
            "requires_crafting_table": recipe.get("requires_crafting_table"),
        }
    if "missing" in payload:
        summary["missing"] = payload["missing"]
    return summary


async def run_smoke(
    name: str,
    call: SmokeCall,
    acceptable_errors: Tuple[str, ...] = (),
) -> Tuple[bool, Dict[str, Any]]:
    try:
        payload = await call()
        print(f"[OK] {name}: {summarize(payload)}", flush=True)
        return True, payload
    except BridgeError as exc:
        if acceptable_errors and any(token in str(exc) for token in acceptable_errors):
            print(f"[OK] {name}: expected unavailable state: {exc}", flush=True)
            return True, {"expected_unavailable": str(exc)}
        print(f"[FAIL] {name}: {exc}", flush=True)
        return False, {"error": str(exc)}
    except (OSError, asyncio.TimeoutError) as exc:
        print(f"[FAIL] {name}: {exc}", flush=True)
        return False, {"error": str(exc)}


async def wait_for_player(client: BridgeClient, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    last_error = ""
    stable_ticks = 0
    while time.monotonic() < deadline:
        try:
            payload = await client.request("get_full_state")
            player = payload.get("player", {})
            if float(player.get("health", 20.0)) <= 0.0 or player.get("is_dead"):
                print(f"[WAIT] player is dead; requesting respawn", flush=True)
                await client.request("respawn")
                stable_ticks = 0
                await asyncio.sleep(2.0)
                continue
            motion = player.get("motion", {})
            vertical_motion = abs(float(motion.get("y", 0.0)))
            if player.get("is_on_ground") and vertical_motion < 0.10:
                stable_ticks += 1
            else:
                stable_ticks = 0
                print(
                    f"[WAIT] player stabilizing: y={player.get('position', {}).get('y')} "
                    f"on_ground={player.get('is_on_ground')} dy={motion.get('y')}",
                    flush=True,
                )
                await asyncio.sleep(1.0)
                continue
            if stable_ticks < 2:
                print("[WAIT] player grounded; confirming stable tick", flush=True)
                await asyncio.sleep(1.0)
                continue
            print(f"[OK] wait_player: {summarize(payload)}", flush=True)
            return True
        except BridgeError as exc:
            last_error = str(exc)
            if "PLAYER_NOT_READY" not in last_error:
                print(f"[WAIT] player not ready: {last_error}", flush=True)
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
            print(f"[WAIT] bridge/player not ready: {last_error}", flush=True)
            await client.ensure_connected()
        await asyncio.sleep(1.0)
    print(f"[FAIL] wait_player: timed out after {timeout:.1f}s; last_error={last_error}", flush=True)
    return False


async def wait_for_recipe_manager(client: BridgeClient, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    last_count = 0
    while time.monotonic() < deadline:
        try:
            payload = await client.request("list_recipes", include_vanilla=True, vanilla_limit=0)
            last_count = int(payload.get("vanilla_crafting_recipes", 0))
            if last_count > 0:
                print(f"[OK] wait_recipes: vanilla_crafting_recipes={last_count}", flush=True)
                return True
            print("[WAIT] recipe manager not ready yet", flush=True)
        except (BridgeError, OSError, asyncio.TimeoutError) as exc:
            print(f"[WAIT] recipe manager not ready: {exc}", flush=True)
        await asyncio.sleep(1.0)
    print(f"[FAIL] wait_recipes: timed out after {timeout:.1f}s; last_count={last_count}", flush=True)
    return False


def inventory_count(inventory: Dict[str, Any], terms: Iterable[str]) -> int:
    normalized_terms = tuple(str(term).lower() for term in terms)
    total = 0
    for item in inventory.get("items", []):
        name = str(item.get("item", "")).lower()
        if any(term in name for term in normalized_terms):
            total += int(item.get("count", 0))
    return total


async def get_inventory(client: BridgeClient) -> Dict[str, Any]:
    payload = await client.request("get_inventory")
    inventory = payload.get("inventory", {})
    return inventory if isinstance(inventory, dict) else {"items": []}


def print_skill(result: SkillResult) -> None:
    marker = "OK" if result.success else "FAIL"
    print(f"[{marker}] progression:{result.name}: reward={result.reward_hint:.2f} details={result.details}", flush=True)


async def ensure_alive(client: BridgeClient, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = await client.request("get_full_state")
        player = payload.get("player", {})
        if float(player.get("health", 20.0)) > 0.0 and not player.get("is_dead"):
            return True
        await client.request("respawn")
        await asyncio.sleep(2.0)
    return False


async def acquire_wood(mechanics: BasicMechanics, client: BridgeClient, minimum_logs: int = 3) -> bool:
    for _ in range(36):
        inventory = await get_inventory(client)
        if (
            inventory_count(inventory, ("log", "wood")) >= minimum_logs
            or inventory_count(inventory, ("planks",)) >= minimum_logs * 4
        ):
            return True

        observation = await mechanics.observe()
        player = observation.get("player", {})
        if float(player.get("health", 20.0)) <= 0.0 or player.get("is_dead"):
            print_skill(await mechanics.run("respawn", observation))
            await asyncio.sleep(1.0)
            continue
        hostile = mechanics.closest_entity(observation, mechanics.is_hostile)
        if hostile and float(hostile.get("distance", 99.0)) < 10.0:
            await handle_hostile(mechanics, observation)
            await asyncio.sleep(0.6)
            continue
        result = await mechanics.skill_harvest_trees(observation)
        print_skill(result)
        if not result.success:
            result = await mechanics.skill_mine_nearest_log(observation)
            print_skill(result)
        if not result.success:
            result = await mechanics.skill_sprint_wander(await mechanics.observe())
            print_skill(result)

        await asyncio.sleep(0.6)
        with suppress(BridgeError):
            collect = await mechanics.skill_collect_item(await mechanics.observe(), ("log", "wood"))
            if not collect.success:
                collect = await mechanics.skill_collect_item(await mechanics.observe())
            if collect.success:
                print_skill(collect)

    inventory = await get_inventory(client)
    return (
        inventory_count(inventory, ("log", "wood")) >= minimum_logs
        or inventory_count(inventory, ("planks",)) >= minimum_logs * 4
    )


async def handle_hostile(mechanics: BasicMechanics, observation: Dict[str, Any]) -> bool:
    hostile = mechanics.closest_entity(observation, mechanics.is_hostile)
    if hostile is None:
        return False

    player = observation.get("player", {})
    health = float(player.get("health", 20.0))
    distance = float(hostile.get("distance", 99.0))
    if health <= 8.0 or distance > 5.0:
        print_skill(await mechanics.skill_flee_hostile(observation))
        return True

    for _ in range(8):
        print_skill(await mechanics.skill_engage_hostile(observation))
        await asyncio.sleep(0.35)
        observation = await mechanics.observe()
        player = observation.get("player", {})
        if float(player.get("health", 20.0)) <= 0.0 or player.get("is_dead"):
            print_skill(await mechanics.run("respawn", observation))
            return True
        hostile = mechanics.closest_entity(observation, mechanics.is_hostile)
        if hostile is None or float(hostile.get("distance", 99.0)) > 8.0:
            return True
        if float(player.get("health", 20.0)) <= 8.0:
            print_skill(await mechanics.skill_flee_hostile(observation))
            return True
    print_skill(await mechanics.skill_flee_hostile(observation))
    return True


async def collect_until_inventory(
    mechanics: BasicMechanics,
    client: BridgeClient,
    terms: Iterable[str],
    minimum: int,
    *,
    attempts: int = 6,
) -> bool:
    for attempt in range(1, attempts + 1):
        inventory = await get_inventory(client)
        count = inventory_count(inventory, terms)
        if count >= minimum:
            return True

        observation = await mechanics.observe()
        collect = await mechanics.skill_collect_item(observation, terms)
        print_skill(collect)
        if collect.success:
            inventory = await get_inventory(client)
            if inventory_count(inventory, terms) >= minimum:
                return True

        await asyncio.sleep(0.35 + attempt * 0.1)

    inventory = await get_inventory(client)
    return inventory_count(inventory, terms) >= minimum


async def run_progression_check(client: BridgeClient) -> bool:
    mechanics = BasicMechanics(client, step_delay=0.2)
    await client.request("set_unpause", state=True)
    alive = await ensure_alive(client)
    print(f"[{'OK' if alive else 'FAIL'}] progression:alive", flush=True)
    if not alive:
        return False
    ok = True

    wood_ok = await acquire_wood(mechanics, client, minimum_logs=2)
    print(f"[{'OK' if wood_ok else 'FAIL'}] progression:wood_acquired", flush=True)
    ok = ok and wood_ok
    if not wood_ok:
        return False

    steps = [
        ("craft_planks", lambda obs: mechanics.skill_craft_planks(obs)),
        ("craft_crafting_table", lambda obs: mechanics.skill_craft_crafting_table(obs)),
        ("open_crafting_table", lambda obs: mechanics.skill_open_crafting_table(obs)),
        ("craft_sticks", lambda obs: mechanics.skill_craft_sticks(obs)),
        ("craft_wooden_pickaxe", lambda obs: mechanics.skill_craft_wooden_pickaxe(obs)),
        ("mine_nearest_stone", lambda obs: mechanics.skill_mine_nearest_stone(obs, desired_count=1)),
    ]

    for name, call in steps:
        observation = await mechanics.observe()
        hostile = mechanics.closest_entity(observation, mechanics.is_hostile)
        if hostile and float(hostile.get("distance", 99.0)) < 10.0:
            await handle_hostile(mechanics, observation)
            await asyncio.sleep(0.5)
            observation = await mechanics.observe()
        result = await call(observation)
        print_skill(result)
        ok = ok and result.success
        if not result.success:
            break
        if name == "mine_nearest_stone":
            collected_stone = await collect_until_inventory(
                mechanics,
                client,
                ("cobblestone", "cobbled_deepslate"),
                1,
                attempts=8,
            )
            print(f"[{'OK' if collected_stone else 'FAIL'}] progression:collect_stone_drop", flush=True)
            ok = ok and collected_stone
        await asyncio.sleep(0.35)

    inventory = await get_inventory(client)
    stone = inventory_count(inventory, ("cobblestone", "cobbled_deepslate"))
    pickaxe = inventory_count(inventory, ("wooden_pickaxe", "stone_pickaxe"))
    print(f"[{'OK' if stone > 0 else 'FAIL'}] progression:stone_inventory count={stone}", flush=True)
    print(f"[{'OK' if pickaxe > 0 else 'FAIL'}] progression:tool_inventory count={pickaxe}", flush=True)
    return ok and stone > 0 and pickaxe > 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the live Minecraft Bridge client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25575)
    parser.add_argument("--auth-token", default="")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--craft-target", default="planks")
    parser.add_argument("--open-world", default="", help="Local singleplayer save to open before world-state checks.")
    parser.add_argument("--wait-player-timeout", type=float, default=90.0)
    parser.add_argument("--progression-check", action="store_true")
    args = parser.parse_args()

    client = BridgeClient(host=args.host, port=args.port, auth_token=args.auth_token, timeout=args.timeout)
    try:
        try:
            await client.connect()
        except (BridgeError, OSError, asyncio.TimeoutError) as exc:
            print(f"[FAIL] connect: {exc}", flush=True)
            return 1
        await run_smoke("ping", lambda: client.request("ping"))
        await run_smoke("get_version", lambda: client.request("get_version"))
        if args.open_world:
            ok, _ = await run_smoke(
                "open_singleplayer_world",
                lambda: client.request("open_singleplayer_world", world=args.open_world),
            )
            if ok:
                player_ready = await wait_for_player(client, args.wait_player_timeout)
                if player_ready:
                    await wait_for_recipe_manager(client, args.wait_player_timeout)
        await run_smoke("set_unpause", lambda: client.request("set_unpause", state=True))
        await run_smoke("set_game_mode_survival", lambda: client.request("set_game_mode", mode="survival"))
        await run_smoke("set_time_day", lambda: client.request("set_time", time="day"))
        await run_smoke("set_difficulty_peaceful", lambda: client.request("set_difficulty", difficulty="peaceful"))
        await asyncio.sleep(2.0)

        checks = [
            ("get_full_state", lambda: client.request("get_full_state")),
            (
                "list_recipes",
                lambda: client.request(
                    "list_recipes",
                    include_vanilla=True,
                    include_details=True,
                    include_plan=True,
                    vanilla_limit=12,
                ),
            ),
            ("query_recipe", lambda: client.request("query_recipe", item=args.craft_target)),
            ("verify_crafting_recipes", lambda: client.request("verify_crafting_recipes", detail_limit=24)),
            ("craft", lambda: client.request("craft", item=args.craft_target, max_crafts=1)),
            (
                "open_crafting_table",
                lambda: client.request("open_crafting_table", radius=8),
                ("CRAFTING_TABLE_NOT_FOUND",),
            ),
            ("get_container", lambda: client.request("get_container")),
            ("get_advancements", lambda: client.request("get_advancements", limit=128)),
            ("get_visual_summary", lambda: client.request("get_visual_summary", rays=9, distance=32.0)),
        ]
        results = []
        for check in checks:
            name, call, *acceptable_errors_list = check
            acceptable_errors = acceptable_errors_list[0] if acceptable_errors_list else ()
            results.append(await run_smoke(name, call, acceptable_errors))
        if args.progression_check:
            await run_smoke("close_container", lambda: client.request("close_container"))
            progression_ok = await run_progression_check(client)
            results.append((progression_ok, {"ok": progression_ok}))
        passed = sum(1 for ok, _ in results if ok)
        print(f"[SUMMARY] passed={passed} failed={len(results) - passed}", flush=True)
        return 0 if passed == len(results) else 1
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
