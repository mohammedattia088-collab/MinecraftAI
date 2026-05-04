import argparse
import asyncio

try:
    from .config import AUTONOMOUS_PARAMS, BRIDGE_AUTH_TOKEN, BRIDGE_HOST, BRIDGE_PORT
    from .training import AutonomousPlayer
    from .bridge import BridgeClient
    from .validation import MultiSeedValidator
except ImportError:
    from config import AUTONOMOUS_PARAMS, BRIDGE_AUTH_TOKEN, BRIDGE_HOST, BRIDGE_PORT
    from training.autonomous_player import AutonomousPlayer
    from bridge import BridgeClient
    from validation import MultiSeedValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous Minecraft learner against the Minecraft Bridge mod.")
    parser.add_argument("--host", default=BRIDGE_HOST)
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    parser.add_argument("--auth-token", default=BRIDGE_AUTH_TOKEN)
    parser.add_argument("--steps", type=int, default=int(AUTONOMOUS_PARAMS["default_steps"]))
    parser.add_argument("--step-delay", type=float, default=float(AUTONOMOUS_PARAMS["step_delay"]))
    parser.add_argument("--goal", default=str(AUTONOMOUS_PARAMS.get("default_goal", "survive and progress")))
    parser.add_argument("--forever", action="store_true", help="Keep training until interrupted.")
    parser.add_argument("--validate-seeds", nargs="*", default=None, help="Run validation across existing worlds named <prefix>_<seed>.")
    parser.add_argument("--validation-steps", type=int, default=120)
    parser.add_argument("--world-prefix", default="AISeed")
    parser.add_argument("--validation-output", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.validate_seeds is not None:
        import json

        seeds = args.validate_seeds or ["0", "1", "2", "3", "4"]
        client = BridgeClient(args.host, args.port, args.auth_token, timeout=float(AUTONOMOUS_PARAMS["request_timeout"]))
        validator = MultiSeedValidator(
            client,
            world_prefix=args.world_prefix,
            steps=args.validation_steps,
            step_delay=args.step_delay,
            goal=args.goal,
        )
        try:
            result = await validator.run(seeds)
        finally:
            await client.close()
        payload = result.as_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.validation_output:
            with open(args.validation_output, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
        return

    player = AutonomousPlayer(
        host=args.host,
        port=args.port,
        auth_token=args.auth_token,
        step_delay=args.step_delay,
        goal=args.goal,
    )
    try:
        await player.run(steps=args.steps, forever=args.forever)
    finally:
        await player.client.close()


if __name__ == "__main__":
    asyncio.run(main())
