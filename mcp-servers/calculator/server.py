"""calculator-mcp — deterministic arithmetic, units and dates.

Models are unreliable at arithmetic in a way that is hard to notice: the answer
looks right, has the right number of digits, and is wrong. This server exists so
the numbers in an answer come from something that cannot be plausibly wrong.

Everything here is deterministic and side-effect free, which is why the client
caches its results for a day.

Expressions are evaluated with SymPy, never with :func:`eval`. That matters more
than it might appear: an expression arrives from a model that may itself be
repeating text from a retrieved document, so it is untrusted input, and
``eval("__import__('os').system(...)")`` is a working remote code execution
against the tool server.

Tools:
    calculate       — evaluate a mathematical expression exactly.
    convert_units   — convert between physical units.
    date_difference — the interval between two dates.
    date_shift      — add or subtract a duration from a date.
    percentage      — percentage change, of, and increase/decrease.

Example:
    >>> import asyncio
    >>> asyncio.run(evaluate_expression("2 + 2 * 3")).content["result"]
    '8'
"""

from __future__ import annotations

import re
from typing import Any

from mcp_common.server import ToolResult, build_app, build_table, require, tool

VERSION = "1.0.0"

#: Characters an expression may contain. Anything else is rejected before SymPy
#: sees it: SymPy's parser is not a sandbox, and its ``sympify`` has historically
#: been able to reach Python builtins.
_ALLOWED_EXPRESSION = re.compile(r"^[0-9a-zA-Z_+\-*/^().,%!<>= \t\[\]]*$")

#: Function names the expression parser will resolve. An unlisted name is an
#: error rather than a silent symbolic variable, because ``os.system`` parsed as
#: a free symbol would evaluate to itself and look like a successful result.
_ALLOWED_FUNCTIONS = frozenset(
    {
        "sqrt", "abs", "exp", "log", "ln", "log10", "sin", "cos", "tan", "asin",
        "acos", "atan", "sinh", "cosh", "tanh", "floor", "ceiling", "round",
        "factorial", "gcd", "lcm", "max", "min", "sum", "pi", "E", "I", "oo",
        "Sum", "Product", "binomial", "Rational", "Integer", "Float",
    }
)  # fmt: skip

#: Guards against an expression that is trivial to write and enormous to compute,
#: such as ``9**9**9``. SymPy would happily try.
MAX_EXPRESSION_LENGTH = 500
MAX_RESULT_DIGITS = 100


async def evaluate_expression(expression: str) -> ToolResult:
    """Evaluate a mathematical expression exactly, then numerically."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return ToolResult.failure(
            f"expression too long ({len(expression)} chars, limit {MAX_EXPRESSION_LENGTH})"
        )
    if not _ALLOWED_EXPRESSION.match(expression):
        return ToolResult.failure(
            "expression contains characters that are not allowed; use numbers, "
            "operators and the supported function names only"
        )

    unknown = _unknown_names(expression)
    if unknown:
        return ToolResult.failure(f"unknown function or name: {', '.join(sorted(unknown))}")

    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            parse_expr,
            standard_transformations,
        )

        parsed = parse_expr(
            expression,
            transformations=(*standard_transformations, convert_xor),
            evaluate=True,
        )
        exact = sympy.simplify(parsed)
        numeric = sympy.N(exact, 15)
    except Exception as exc:  # noqa: BLE001 - a bad expression is a failed result
        return ToolResult.failure(f"could not evaluate: {exc}")

    # SymPy returns `zoo` for 1/0 and `nan` for 0/0 rather than raising. Handing
    # either back as a result would be quietly wrong: the honest answer to "what
    # is 1/0" is that it is undefined.
    if str(exact) in ("zoo", "nan", "oo", "-oo") or str(numeric) in ("zoo", "nan"):
        return ToolResult.failure(f"{expression} is undefined or not finite")

    exact_text = str(exact)
    numeric_text = str(numeric)
    if len(exact_text) > MAX_RESULT_DIGITS:
        exact_text = exact_text[:MAX_RESULT_DIGITS] + "..."

    return ToolResult.success(
        {"expression": expression, "result": exact_text, "numeric": numeric_text},
        summary=f"{expression} = {exact_text}",
    )


def _unknown_names(expression: str) -> set[str]:
    """Identifiers in the expression that are not allowed functions.

    Single letters are treated as symbolic variables, which is legitimate for an
    algebraic expression; longer names must be recognised.

    Example:
        >>> sorted(_unknown_names("sqrt(x) + system(1)"))
        ['system']
        >>> _unknown_names("2 * x + sqrt(4)")
        set()
    """
    names = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression))
    return {n for n in names if len(n) > 1 and n not in _ALLOWED_FUNCTIONS}


async def convert(value: float, from_unit: str, to_unit: str) -> ToolResult:
    """Convert a quantity between units."""
    try:
        import pint

        registry = pint.UnitRegistry()
        quantity = registry.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
    except Exception as exc:  # noqa: BLE001 - an impossible conversion is a result
        return ToolResult.failure(f"could not convert {from_unit} to {to_unit}: {exc}")

    return ToolResult.success(
        {
            "value": float(converted.magnitude),
            "unit": str(converted.units),
            "original": {"value": value, "unit": from_unit},
        },
        summary=f"{value} {from_unit} = {converted.magnitude:.6g} {to_unit}",
    )


async def date_difference(start: str, end: str, unit: str = "days") -> ToolResult:
    """Compute the interval between two dates."""
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta

    try:
        first = date_parser.parse(start)
        second = date_parser.parse(end)
    except (ValueError, OverflowError) as exc:
        return ToolResult.failure(f"could not parse dates: {exc}")

    delta = second - first
    relative = relativedelta(second, first)
    values = {
        "days": delta.days,
        "hours": delta.total_seconds() / 3600,
        "weeks": delta.days / 7,
        "months": relative.years * 12 + relative.months,
        "years": relative.years,
    }
    if unit not in values:
        return ToolResult.failure(f"unknown unit {unit!r}; use one of {', '.join(values)}")

    return ToolResult.success(
        {
            "value": values[unit],
            "unit": unit,
            "start": first.isoformat(),
            "end": second.isoformat(),
            "all_units": values,
        },
        summary=f"{values[unit]} {unit} between {first.date()} and {second.date()}",
    )


async def date_shift(date: str, amount: int, unit: str = "days") -> ToolResult:
    """Add or subtract a duration from a date."""
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta

    try:
        origin = date_parser.parse(date)
    except (ValueError, OverflowError) as exc:
        return ToolResult.failure(f"could not parse date: {exc}")

    if unit not in ("days", "weeks", "months", "years", "hours"):
        return ToolResult.failure(f"unknown unit {unit!r}")

    shifted = origin + relativedelta(**{unit: amount})
    return ToolResult.success(
        {"result": shifted.isoformat(), "original": origin.isoformat()},
        summary=f"{origin.date()} {amount:+d} {unit} = {shifted.date()}",
    )


async def percentage(operation: str, a: float, b: float) -> ToolResult:
    """Percentage change, percentage-of, and increase/decrease.

    Broken out as its own tool because percentage change is the single
    calculation models get wrong most often, usually by dividing by the wrong
    operand.
    """
    operations = {
        "change": lambda: ((b - a) / a * 100) if a else None,
        "of": lambda: a * b / 100,
        "increase": lambda: a * (1 + b / 100),
        "decrease": lambda: a * (1 - b / 100),
    }
    if operation not in operations:
        return ToolResult.failure(
            f"unknown operation {operation!r}; use one of {', '.join(operations)}"
        )

    result = operations[operation]()
    if result is None:
        return ToolResult.failure("percentage change from zero is undefined")

    labels = {
        "change": f"{a} to {b} is a change of {result:.4g}%",
        "of": f"{b}% of {a} is {result:.6g}",
        "increase": f"{a} increased by {b}% is {result:.6g}",
        "decrease": f"{a} decreased by {b}% is {result:.6g}",
    }
    return ToolResult.success({"result": result, "operation": operation}, summary=labels[operation])


# ── Tool wiring ──────────────────────────────────────────────────────────────


@tool(
    "calculate",
    "Evaluate a mathematical expression exactly. Use this for any arithmetic "
    "rather than computing it yourself.",
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. '(1200 * 1.08) / 12' or 'sqrt(2) + pi'",
            }
        },
        "required": ["expression"],
    },
    deterministic=True,
)
async def calculate_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Evaluate an expression."""
    return await evaluate_expression(require(arguments, "expression"))


@tool(
    "convert_units",
    "Convert a value between physical units, e.g. miles to km, GB to MB, C to F.",
    {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string"},
            "to_unit": {"type": "string"},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    deterministic=True,
)
async def convert_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Convert between units."""
    return await convert(
        float(require(arguments, "value", (int, float))),  # type: ignore[arg-type]
        require(arguments, "from_unit"),
        require(arguments, "to_unit"),
    )


@tool(
    "date_difference",
    "The interval between two dates, in days, weeks, months or years.",
    {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "end": {"type": "string"},
            "unit": {"type": "string", "enum": ["days", "weeks", "months", "years", "hours"]},
        },
        "required": ["start", "end"],
    },
    deterministic=True,
)
async def date_difference_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Compute a date interval."""
    return await date_difference(
        require(arguments, "start"), require(arguments, "end"), arguments.get("unit", "days")
    )


@tool(
    "date_shift",
    "Add or subtract a duration from a date. Use a negative amount to subtract.",
    {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "amount": {"type": "integer"},
            "unit": {"type": "string", "enum": ["days", "weeks", "months", "years", "hours"]},
        },
        "required": ["date", "amount"],
    },
    deterministic=True,
)
async def date_shift_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Shift a date."""
    return await date_shift(
        require(arguments, "date"),
        int(require(arguments, "amount", int)),
        arguments.get("unit", "days"),
    )


@tool(
    "percentage",
    "Percentage change between two values, a percentage of a value, or a value "
    "increased or decreased by a percentage.",
    {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["change", "of", "increase", "decrease"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
    },
    deterministic=True,
)
async def percentage_handler(arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Compute a percentage."""
    return await percentage(
        require(arguments, "operation"),
        float(require(arguments, "a", (int, float))),  # type: ignore[arg-type]
        float(require(arguments, "b", (int, float))),  # type: ignore[arg-type]
    )


TOOLS = build_table(
    calculate_handler,
    convert_handler,
    date_difference_handler,
    date_shift_handler,
    percentage_handler,
)

app = build_app(
    name="calculator",
    version=VERSION,
    tools=TOOLS,
    description=__doc__ or "",
)
