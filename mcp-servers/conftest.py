"""Shared fixtures for the MCP server tests.

The servers live in sibling directories that are not importable packages — each
is a standalone deployable — so the loader below imports each ``server.py`` by
path with its own directory on ``sys.path``, which is exactly how the container
runs it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "common"))


def load_server(name: str) -> ModuleType:
    """Import a server module by directory name.

    Registers the module in ``sys.modules`` before executing it, because
    dataclasses resolve their annotations by looking the defining module up
    there — omitting that produces a confusing ``NoneType has no __dict__``.
    """
    directory = ROOT / name
    module_name = f"{name.replace('-', '_')}_server"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, directory / "server.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load server module for {name}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


@pytest.fixture(scope="session")
def calculator() -> ModuleType:
    """The calculator server module."""
    return load_server("calculator")


@pytest.fixture(scope="session")
def code_exec() -> ModuleType:
    """The code-exec server module."""
    return load_server("code-exec")


@pytest.fixture(scope="session")
def sql_analytics() -> ModuleType:
    """The sql-analytics server module."""
    return load_server("sql-analytics")


@pytest.fixture(scope="session")
def web_fetch() -> ModuleType:
    """The web-fetch server module."""
    return load_server("web-fetch")


@pytest.fixture(scope="session")
def docs_search() -> ModuleType:
    """The docs-search server module."""
    return load_server("docs-search")


@pytest.fixture(scope="session")
def kg_query() -> ModuleType:
    """The kg-query server module."""
    return load_server("kg-query")
