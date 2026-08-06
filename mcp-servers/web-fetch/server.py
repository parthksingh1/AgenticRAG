"""web-fetch-mcp — allowlisted web fetching, converted to markdown.

The corrective-RAG fallback needs the open web when a tenant's documents cannot
answer a question. Fetching arbitrary URLs on behalf of a model is a
server-side request forgery primitive, so the constraint is not "be careful"
but "the model cannot express a request that reaches anywhere interesting".

Four layers:

* **A domain allowlist**, configured per deployment. Not a denylist: the set of
  interesting internal hostnames is unbounded and grows with the infrastructure.
* **DNS resolution before connecting**, with every resolved address checked
  against the private, loopback, link-local and reserved ranges. Checking the
  hostname alone is defeated by a public name with an ``A`` record pointing at
  ``169.254.169.254`` — the cloud metadata endpoint — which is the standard SSRF
  escalation.
* **Redirects followed manually**, revalidating each hop. Following redirects
  automatically lets an allowlisted domain bounce the request anywhere.
* **robots.txt respected**, and a size and time ceiling on the response.

Example:
    >>> is_private_address("169.254.169.254")
    True
    >>> is_private_address("93.184.216.34")
    False
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

from mcp_common.server import ToolResult, build_app, build_table, require, tool

VERSION = "1.0.0"

#: Domains this deployment may fetch. Read from the environment so it is a
#: deployment decision rather than a code change. Subdomains are included.
DEFAULT_ALLOWLIST = (
    "arxiv.org",
    "en.wikipedia.org",
    "docs.python.org",
    "developer.mozilla.org",
    "www.postgresql.org",
    "opensearch.org",
)

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 10.0
MAX_REDIRECTS = 3
USER_AGENT = "AgenticRAG-web-fetch/1.0 (+https://github.com/agrag)"


def allowlist() -> frozenset[str]:
    """Domains permitted for this deployment.

    Example:
        >>> "arxiv.org" in allowlist()
        True
    """
    configured = os.getenv("WEB_FETCH_ALLOWLIST", "")
    if configured.strip():
        return frozenset(d.strip().lower() for d in configured.split(",") if d.strip())
    return frozenset(DEFAULT_ALLOWLIST)


def is_private_address(address: str) -> bool:
    """Whether an IP is in a range that must never be fetched.

    Covers loopback, private, link-local (including the cloud metadata address),
    multicast and reserved space.

    Example:
        >>> [is_private_address(a) for a in ("127.0.0.1", "10.0.0.1", "169.254.169.254")]
        [True, True, True]
        >>> is_private_address("8.8.8.8")
        False
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True  # unparseable is not a reason to proceed
    return (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


@dataclass(frozen=True, slots=True)
class UrlVerdict:
    """The outcome of validating a URL."""

    ok: bool
    reason: str | None = None
    host: str | None = None


def validate_url(url: str, *, resolve: bool = True) -> UrlVerdict:
    """Check scheme, host, allowlist and every resolved address.

    Args:
        url: The URL to check.
        resolve: Whether to resolve DNS. Disabled in tests that must not touch
            the network; production always resolves, because the allowlist alone
            does not stop a DNS rebind to a metadata endpoint.

    Example:
        >>> validate_url("http://localhost/admin", resolve=False).reason
        'host is not on the allowlist: localhost'
        >>> validate_url("ftp://arxiv.org/x", resolve=False).reason
        'only http and https are supported, got ftp'
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return UrlVerdict(
            ok=False, reason=f"only http and https are supported, got {parsed.scheme}"
        )

    host = (parsed.hostname or "").lower()
    if not host:
        return UrlVerdict(ok=False, reason="URL has no host")

    permitted = allowlist()
    if not any(host == domain or host.endswith(f".{domain}") for domain in permitted):
        return UrlVerdict(ok=False, reason=f"host is not on the allowlist: {host}")

    if resolve:
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            return UrlVerdict(ok=False, reason=f"could not resolve {host}: {exc}")

        for info in infos:
            address = info[4][0]
            if is_private_address(str(address)):
                # An allowlisted name resolving into private space is the
                # rebinding attack, not a misconfiguration.
                return UrlVerdict(
                    ok=False,
                    reason=f"{host} resolves to a non-public address and will not be fetched",
                )

    return UrlVerdict(ok=True, host=host)


async def robots_allows(url: str, client: Any) -> bool:
    """Whether robots.txt permits fetching this URL.

    A robots.txt that cannot be fetched is treated as permissive, which matches
    the convention: sites that care serve one, and treating a 404 as a refusal
    would make the tool useless on most of the web.
    """
    from urllib.robotparser import RobotFileParser

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = await client.get(robots_url, timeout=5.0)
        if response.status_code != 200:
            return True
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
    except Exception:  # noqa: BLE001 - unreachable robots.txt is permissive
        return True
    return bool(parser.can_fetch(USER_AGENT, url))


async def fetch(url: str, *, respect_robots: bool = True) -> ToolResult:
    """Fetch a URL and return its main content as markdown."""
    import httpx

    verdict = validate_url(url)
    if not verdict.ok:
        return ToolResult.failure(f"refused to fetch: {verdict.reason}")

    async with httpx.AsyncClient(
        timeout=TIMEOUT_SECONDS,
        follow_redirects=False,  # each hop is revalidated by hand
        headers={"User-Agent": USER_AGENT},
    ) as client:
        if respect_robots and not await robots_allows(url, client):
            return ToolResult.failure(f"robots.txt disallows fetching {url}")

        current = url
        for _ in range(MAX_REDIRECTS + 1):
            try:
                response = await client.get(current)
            except Exception as exc:  # noqa: BLE001 - a fetch failure is a result
                return ToolResult.failure(f"fetch failed: {exc}")

            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    return ToolResult.failure("redirect without a location header")
                current = urljoin(current, location)
                hop = validate_url(current)
                if not hop.ok:
                    # The interesting case: an allowlisted domain redirecting
                    # somewhere it should not be able to reach.
                    return ToolResult.failure(f"refused to follow redirect: {hop.reason}")
                continue

            if response.status_code >= 400:
                return ToolResult.failure(f"HTTP {response.status_code} from {current}")

            content_type = response.headers.get("content-type", "")
            if not any(t in content_type for t in ("text/html", "text/plain", "xml", "json")):
                return ToolResult.failure(f"unsupported content type: {content_type or 'unknown'}")

            body = response.content[:MAX_RESPONSE_BYTES]
            markdown = to_markdown(body.decode("utf-8", errors="replace"), url=current)
            return ToolResult.success(
                {
                    "url": current,
                    "title": extract_title(response.text),
                    "content": markdown,
                    "truncated": len(response.content) > MAX_RESPONSE_BYTES,
                },
                summary=f"fetched {len(markdown)} characters from {verdict.host}",
                metadata={"status": response.status_code, "content_type": content_type},
            )

        return ToolResult.failure(f"too many redirects (limit {MAX_REDIRECTS})")


def to_markdown(html: str, *, url: str) -> str:
    """Extract the main content of a page as markdown.

    Uses trafilatura, falling back to a plain tag strip. The fallback matters
    because this runs in the corrective path: returning nothing here means the
    user gets "I could not find an answer" when an answer was available.
    """
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, output_format="markdown", include_tables=True, url=url
        )
        if extracted:
            return extracted
    except Exception as exc:  # noqa: BLE001 - fall through to the simple extractor
        logging.getLogger(__name__).warning("trafilatura failed for %s: %s", url, exc)

    import html as html_module
    import re

    text = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def extract_title(html: str) -> str | None:
    """The page title, if it has one.

    Example:
        >>> extract_title("<html><title>Hello</title></html>")
        'Hello'
        >>> extract_title("<html></html>") is None
        True
    """
    import re

    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    import html as html_module

    return html_module.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip() or None


@tool(
    "fetch_url",
    "Fetch a web page from an allowlisted domain and return its main content as "
    "markdown. Use this only when the workspace documents cannot answer.",
    {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "An http or https URL."}},
        "required": ["url"],
    },
    deterministic=False,
    read_only=True,
)
async def fetch_url_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Fetch a URL."""
    return await fetch(require(arguments, "url"))


@tool(
    "list_allowed_domains",
    "List the domains this deployment is permitted to fetch from.",
    {"type": "object", "properties": {}},
    deterministic=True,
    read_only=True,
)
async def list_domains_handler(_arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """List the allowlisted domains."""
    domains = sorted(allowlist())
    return ToolResult.success({"domains": domains}, summary=f"{len(domains)} allowed domains")


TOOLS = build_table(fetch_url_handler, list_domains_handler)

app = build_app(
    name="web-fetch",
    version=VERSION,
    tools=TOOLS,
    description=__doc__ or "",
)
