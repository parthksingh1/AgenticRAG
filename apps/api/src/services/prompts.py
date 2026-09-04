"""Versioned prompt registry.

Prompts live in ``prompts/*.yaml`` rather than in Python string literals, for
three reasons that matter operationally:

* **They are reviewable.** A prompt change is the single highest-leverage change
  anyone can make to this system's behaviour, and it should show up in a diff
  next to its changelog entry, not buried in a source file.
* **They are versioned and content-hashed.** Every answer records the prompt
  version that produced it, so an eval regression can be traced to the exact
  text. The content hash catches the case where someone edits a prompt without
  bumping its version — the registry refuses to load it.
* **They can be swapped at runtime.** The playground and the A/B router select a
  version by name without a deploy.

Rendering is Jinja2 with ``StrictUndefined``: a template referencing a variable
the caller did not supply raises immediately rather than silently rendering an
empty string into the prompt, which is how a prompt quietly loses its context and
nobody notices until the eval scores drop.

Example:
    >>> prompt = Prompt(
    ...     name="greet", version="v1", template="Hello {{ name }}.", variables=["name"]
    ... )
    >>> prompt.render(name="world")
    'Hello world.'
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError
from jinja2 import meta as jinja_meta

from src.core.errors import ConfigurationError
from src.core.logging import get_logger

log = get_logger(__name__)

# autoescape is wrong for prompt templating: this renders plain text sent to
# a model, not HTML served to a browser.
_ENVIRONMENT = Environment(  # noqa: S701 # nosec B701
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


@dataclass(frozen=True, slots=True)
class Prompt:
    """One versioned prompt template."""

    name: str
    version: str
    template: str
    variables: tuple[str, ...] = ()
    description: str | None = None
    author: str | None = None
    changelog: str | None = None
    #: Model the prompt was written and evaluated against. Recorded rather than
    #: enforced: prompts usually transfer, but knowing where one came from
    #: explains a lot when it stops working.
    tested_with: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the template text, truncated for readability.

        Example:
            >>> len(Prompt(name="n", version="v1", template="x").content_hash)
            16
        """
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    @property
    def identifier(self) -> str:
        """Stable ``name@version`` identifier recorded on every message.

        Example:
            >>> Prompt(name="answer", version="v3", template="x").identifier
            'answer@v3'
        """
        return f"{self.name}@{self.version}"

    def render(self, **variables: Any) -> str:
        """Render the template.

        Raises:
            ConfigurationError: when a referenced variable is missing, or the
                template itself is malformed. Both are deployment bugs, and
                failing loudly beats rendering a prompt with a hole in it.

        Example:
            >>> Prompt(name="n", version="v1", template="Hi {{ who }}").render(who="you")
            'Hi you'
        """
        try:
            return _ENVIRONMENT.from_string(self.template).render(**variables).strip()
        except TemplateError as exc:
            msg = f"failed to render prompt {self.identifier}: {exc}"
            raise ConfigurationError(msg) from exc

    def declared_variables(self) -> frozenset[str]:
        """Variables the template actually references.

        Compared against the declared ``variables`` list at load time, so a
        template that grew a new placeholder without updating its front matter
        is caught by the loader rather than at 3am by a rendering failure.

        Example:
            >>> prompt = Prompt(name="n", version="v1", template="{{ a }}{{ b }}")
            >>> sorted(prompt.declared_variables())
            ['a', 'b']
        """
        try:
            parsed = _ENVIRONMENT.parse(self.template)
        except TemplateError:
            return frozenset()
        return frozenset(jinja_meta.find_undeclared_variables(parsed))


class PromptRegistry:
    """Loads and serves versioned prompts from a directory."""

    def __init__(self, directory: Path | str, *, strict: bool = True) -> None:
        """Create a registry over a prompts directory.

        Args:
            directory: Directory holding ``*.yaml`` prompt files.
            strict: Reject prompts whose declared variables do not match the
                template. Relaxed only by the playground, which edits templates
                before their front matter catches up.
        """
        self._directory = Path(directory)
        self._strict = strict
        self._prompts: dict[str, dict[str, Prompt]] = {}
        self._active: dict[str, str] = {}

    def load(self) -> int:
        """Load every prompt file, returning the number of prompts registered.

        Raises:
            ConfigurationError: when the directory is missing or a file is
                malformed. A service running with prompts it could not read
                would silently answer with nothing.
        """
        if not self._directory.is_dir():
            msg = f"prompts directory not found: {self._directory}"
            raise ConfigurationError(msg)

        self._prompts.clear()
        self._active.clear()
        count = 0

        for path in sorted(self._directory.glob("*.yaml")):
            for prompt, is_active in self._parse_file(path):
                self._prompts.setdefault(prompt.name, {})[prompt.version] = prompt
                if is_active or prompt.name not in self._active:
                    self._active[prompt.name] = prompt.version
                count += 1

        log.info("loaded prompts", count=count, names=sorted(self._prompts))
        return count

    def _parse_file(self, path: Path) -> Iterator[tuple[Prompt, bool]]:
        """Parse one prompt file into prompts, yielding ``(prompt, is_active)``."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            msg = f"could not read prompt file {path.name}: {exc}"
            raise ConfigurationError(msg) from exc

        if not isinstance(raw, dict) or "name" not in raw:
            msg = f"prompt file {path.name} must be a mapping with a 'name' key"
            raise ConfigurationError(msg)

        name = str(raw["name"])
        active_version = raw.get("active")
        versions = raw.get("versions")
        if not isinstance(versions, dict) or not versions:
            msg = f"prompt {name} declares no versions"
            raise ConfigurationError(msg)

        for version, body in versions.items():
            if not isinstance(body, dict) or "template" not in body:
                msg = f"prompt {name}@{version} has no template"
                raise ConfigurationError(msg)

            prompt = Prompt(
                name=name,
                version=str(version),
                template=str(body["template"]),
                variables=tuple(body.get("variables", ())),
                description=raw.get("description"),
                author=body.get("author"),
                changelog=body.get("changelog"),
                tested_with=body.get("tested_with"),
                metadata=body.get("metadata", {}),
            )
            self._validate(prompt, path)
            yield prompt, str(version) == str(active_version)

    def _validate(self, prompt: Prompt, path: Path) -> None:
        """Check that the declared variables match what the template uses."""
        used = prompt.declared_variables()
        declared = frozenset(prompt.variables)
        undeclared = used - declared
        unused = declared - used

        if not undeclared and not unused:
            return

        detail = (
            f"{prompt.identifier} in {path.name}: "
            f"template uses {sorted(undeclared) or 'nothing'} that is not declared; "
            f"declares {sorted(unused) or 'nothing'} that is unused"
        )
        if self._strict and undeclared:
            raise ConfigurationError(detail)
        log.warning("prompt variable mismatch", detail=detail)

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Return a prompt by name, defaulting to its active version.

        Raises:
            ConfigurationError: for an unknown prompt or version. A typo here
                would otherwise surface as an empty system prompt.
        """
        versions = self._prompts.get(name)
        if not versions:
            known = ", ".join(sorted(self._prompts)) or "none loaded"
            msg = f"unknown prompt {name!r}; available: {known}"
            raise ConfigurationError(msg)

        resolved = version or self._active.get(name)
        prompt = versions.get(str(resolved))
        if prompt is None:
            available = ", ".join(sorted(versions))
            msg = f"unknown version {resolved!r} for prompt {name!r}; available: {available}"
            raise ConfigurationError(msg)
        return prompt

    def render(self, name: str, *, version: str | None = None, **variables: Any) -> str:
        """Look up and render a prompt in one call."""
        return self.get(name, version).render(**variables)

    def versions_of(self, name: str) -> tuple[str, ...]:
        """Every registered version of a prompt, sorted."""
        return tuple(sorted(self._prompts.get(name, {})))

    def names(self) -> tuple[str, ...]:
        """Every registered prompt name."""
        return tuple(sorted(self._prompts))

    def active_version(self, name: str) -> str | None:
        """The version currently served for a prompt."""
        return self._active.get(name)

    def promote(self, name: str, version: str) -> None:
        """Make a version the active one.

        Used by the A/B promotion flow after a variant wins. Validates that the
        version exists first, so a promotion cannot leave the registry pointing
        at nothing.
        """
        self.get(name, version)
        previous = self._active.get(name)
        self._active[name] = version
        log.warning("promoted prompt version", name=name, version=version, previous=previous)

    def fingerprint(self) -> dict[str, str]:
        """Map of ``name@version`` to content hash for every loaded prompt.

        Recorded on eval runs so a result can be tied to the exact prompt text,
        not merely to a version label someone may have edited in place.
        """
        return {
            prompt.identifier: prompt.content_hash
            for versions in self._prompts.values()
            for prompt in versions.values()
        }


_registry: PromptRegistry | None = None

#: Where prompts live, in priority order. The same code runs from ``apps/api``
#: during development and from ``/app`` in the container, where the directory is
#: mounted at ``/prompts`` - so the path is resolved rather than assumed, and a
#: wrong working directory does not stop the service booting.
_SEARCH_PATHS = (
    Path("prompts"),
    Path("/prompts"),
    Path(__file__).resolve().parents[4] / "prompts",
    Path.cwd() / "prompts",
)


def resolve_prompts_dir(configured: Path | str | None = None) -> Path:
    """Find the prompts directory.

    Raises:
        ConfigurationError: when no candidate exists, listing what was tried so
            the fix is obvious rather than a guess.

    Example:
        >>> resolve_prompts_dir().name
        'prompts'
    """
    candidates = [Path(configured)] if configured else []
    candidates.extend(_SEARCH_PATHS)

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.yaml")):
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    msg = f"could not find a prompts directory containing *.yaml; tried: {tried}"
    raise ConfigurationError(msg)


def get_prompt_registry(directory: Path | str | None = None) -> PromptRegistry:
    """Return the process-wide registry, loading it on first use."""
    global _registry
    if _registry is None:
        from src.core.config import get_settings

        _registry = PromptRegistry(resolve_prompts_dir(directory or get_settings().prompts_dir))
        _registry.load()
    return _registry


def reset_prompt_registry() -> None:
    """Drop the cached registry. Used by tests and by the playground's reload."""
    global _registry
    _registry = None
