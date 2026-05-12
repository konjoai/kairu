"""``kairu`` console-script entry point.

Subcommands
-----------
``serve``  Run the SSE inference server::

    kairu serve --model mock --host 0.0.0.0 --port 8000 \
                --cache-capacity 256 --adaptive-gamma \
                --rate-limit 30 --rate-window 10 \
                --redis redis://localhost:6379/0

``bench``  Re-export of :func:`kairu.bench.main` for convenience.

``version``  Print the installed kairu version.

The server subcommand wires :func:`kairu.server.create_app` against the
optional flags. ``--model mock`` runs against :class:`kairu.MockModel` and
needs zero ML dependencies — useful for smoke tests, Docker images, and
anyone benchmarking the orchestration layer in isolation.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from kairu import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kairu",
        description="Real-time LLM inference optimizer — CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sv = sub.add_parser("serve", help="Run the SSE inference server.")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument(
        "--model", default="mock",
        help="Model spec. 'mock' uses MockModel (no ML deps). "
             "Any other value is passed to wrap_model() — HuggingFace model id.",
    )
    sv.add_argument("--model-name", default=None,
                    help="Logical name advertised in /generate responses.")
    sv.add_argument("--cache-capacity", type=int, default=0,
                    help="LogitsCache slots (0 disables caching). Wraps the model.")
    sv.add_argument("--adaptive-gamma", action="store_true",
                    help="Wire a DynamicGammaScheduler into speculative decoding.")
    sv.add_argument("--max-prompt-chars", type=int, default=8192)
    sv.add_argument("--max-tokens-cap", type=int, default=512)
    sv.add_argument("--request-timeout", type=float, default=30.0)
    sv.add_argument("--rate-limit", type=int, default=10,
                    help="Max requests per window (per client IP).")
    sv.add_argument("--rate-window", type=float, default=10.0,
                    help="Sliding-window length in seconds.")
    sv.add_argument("--redis", default=None,
                    help="redis:// URL — switch the rate-limiter backend "
                         "to RedisBackend (requires the redis package).")
    sv.add_argument("--log-level", default="info",
                    choices=["critical", "error", "warning", "info", "debug"])

    sub.add_parser("bench", help="Run the BenchmarkRunner CLI (see kairu.bench).")
    sub.add_parser("version", help="Print version and exit.")

    sh = sub.add_parser("shield", help="Screen a prompt through the default PromptShield.")
    sh.add_argument("prompt", help="Prompt text to screen.")
    sh.add_argument("--json", dest="as_json", action="store_true",
                    help="Output raw JSON instead of human-readable text.")
    return p


def _build_model(spec: str):
    """Resolve a model spec to a ModelInterface instance."""
    if spec == "mock":
        from kairu.mock_model import MockModel
        return MockModel()
    # Anything else: try HuggingFace, fall back to mock for resilience.
    from kairu.wrapper import wrap_model
    return wrap_model(spec).model


async def _build_redis_backend(url: str):
    """Construct a RedisBackend from a redis:// URL. Lazy import."""
    try:
        import redis.asyncio as aioredis  # type: ignore
    except ImportError as e:  # pragma: no cover — surfaces only when --redis used
        raise SystemExit(
            "kairu serve --redis requires the 'redis' package. "
            "Install with: pip install 'redis>=5.0'"
        ) from e
    from kairu.rate_limit import RedisBackend
    client = aioredis.from_url(url, decode_responses=True)
    return RedisBackend(client)


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "kairu serve requires uvicorn. Install with: pip install 'kairu[server]'"
        ) from e

    from kairu.server import ServerConfig, create_app

    cfg = ServerConfig(
        model_name=args.model_name or args.model,
        max_prompt_chars=args.max_prompt_chars,
        max_tokens_cap=args.max_tokens_cap,
        request_timeout_s=args.request_timeout,
        rate_limit_requests=args.rate_limit,
        rate_limit_window_s=args.rate_window,
    )

    backend = None
    if args.redis:
        import asyncio
        backend = asyncio.get_event_loop().run_until_complete(
            _build_redis_backend(args.redis)
        )

    base_model = _build_model(args.model)
    if args.cache_capacity > 0:
        from kairu.kv_cache import CachedModel
        base_model = CachedModel(base_model, cache_capacity=args.cache_capacity)

    app = create_app(model=base_model, config=cfg, rate_limit_backend=backend)

    print(
        f"\n  \033[1;36m流  K A I R U   S E R V E\033[0m  v{__version__}\n"
        f"  model            {args.model}\n"
        f"  cache capacity   {args.cache_capacity}\n"
        f"  adaptive gamma   {args.adaptive_gamma}\n"
        f"  rate limit       {args.rate_limit} req / {args.rate_window}s "
        f"({'redis' if backend else 'in-memory'})\n"
        f"  listening on     \033[1;32mhttp://{args.host}:{args.port}\033[0m\n"
        f"  metrics at       \033[1;32mhttp://{args.host}:{args.port}/metrics\033[0m\n"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_bench(_args: argparse.Namespace) -> int:
    from kairu.bench import main as bench_main
    bench_main()
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    """Print the installed kairu version and exit."""
    print(f"kairu {__version__}")
    return 0


def cmd_shield(args: argparse.Namespace) -> int:
    """Screen a prompt through the default PromptShield and print the result."""
    from kairu.shield import get_default_shield

    result = get_default_shield().check(args.prompt)
    if args.as_json:
        payload = {
            "verdict": str(result.verdict),
            "reason": result.reason,
            "confidence": result.confidence,
            "matched_rule": result.matched_rule,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"verdict      : {result.verdict}")
        print(f"reason       : {result.reason}")
        print(f"confidence   : {result.confidence:.3f}")
        print(f"matched_rule : {result.matched_rule}")
    return 0 if str(result.verdict) == "allowed" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch CLI subcommands: serve, bench, version, shield."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "serve": cmd_serve,
        "bench": cmd_bench,
        "version": cmd_version,
        "shield": cmd_shield,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
