"""Tests for kairu.cli — argparse + dispatch + version."""
from __future__ import annotations

import io
import sys

import pytest

from kairu import __version__
from kairu.cli import _build_parser, cmd_version, main


def test_parser_recognizes_subcommands():
    p = _build_parser()
    assert p.parse_args(["version"]).command == "version"
    assert p.parse_args(["bench"]).command == "bench"
    args = p.parse_args(["serve", "--port", "9001"])
    assert args.command == "serve"
    assert args.port == 9001


def test_parser_serve_defaults():
    args = _build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.model == "mock"
    assert args.cache_capacity == 0
    assert args.adaptive_gamma is False
    assert args.rate_limit == 10
    assert args.rate_window == 10.0


def test_parser_serve_full_overrides():
    argv = [
        "serve", "--host", "0.0.0.0", "--port", "9000",
        "--model", "mock", "--cache-capacity", "128",
        "--adaptive-gamma", "--max-prompt-chars", "2048",
        "--max-tokens-cap", "256", "--request-timeout", "5",
        "--rate-limit", "30", "--rate-window", "60",
    ]
    args = _build_parser().parse_args(argv)
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.cache_capacity == 128
    assert args.adaptive_gamma is True
    assert args.rate_limit == 30
    assert args.rate_window == 60.0


def test_version_subcommand_prints_version(capsys):
    rc = main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert __version__ in captured.out


def test_main_routes_to_version():
    rc = cmd_version(None)  # type: ignore[arg-type]
    assert rc == 0


def test_parser_requires_subcommand():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_parser_rejects_unknown_subcommand():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["nope"])
