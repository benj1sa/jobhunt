"""Token and cost accounting for every model call.

Clients report each call to a shared `UsageMeter`. Commands open a step with `meter.step("extract")`, and when
the step ends its totals are printed and saved to `llm_usage`, which `jobhunt status` reads to estimate costs.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger

from sqlalchemy.ext.asyncio import AsyncSession

from ..models.pipeline import LlmUsage

logger = getLogger(__name__)

# USD per million tokens (input, output), used only when the provider doesn't report cost.
# OpenRouter reports actual cost on every response, so these mainly matter for Jev and for estimates.
LIST_PRICES: dict[str, tuple[float, float]] = {
    "jev": (0.042, 0.0),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "openai/gpt-6.1-terra": (1.00, 5.00),
}
FALLBACK_PRICE = (1.00, 5.00)  # deliberately high for unknown models, so estimates err on the safe side


def list_price(model: str) -> tuple[float, float]:
    if model.startswith("chatgpt/"):
        return (0.0, 0.0)  # runs on the ChatGPT plan's allowance, not a per-token bill
    if model in LIST_PRICES:
        return LIST_PRICES[model]
    if model.startswith("jev"):
        return LIST_PRICES["jev"]
    return FALLBACK_PRICE


def price(model: str, input_tokens: float, output_tokens: float) -> float:
    per_in, per_out = list_price(model)
    return (input_tokens * per_in + output_tokens * per_out) / 1_000_000


@dataclass
class StepUsage:
    stage: str
    provider: str
    model: str
    calls: int = 0
    items: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost: float = 0.0
    calls_with_cost: int = 0

    @property
    def cost(self) -> float:
        """Reported cost where the provider gave one, list-price estimate for the remaining calls."""
        if self.calls_with_cost == self.calls:
            return self.reported_cost
        missing = 1 - (self.calls_with_cost / self.calls if self.calls else 0)
        return self.reported_cost + price(self.model, self.input_tokens * missing, self.output_tokens * missing)

    @property
    def estimated(self) -> bool:
        return self.calls_with_cost < self.calls


class UsageMeter:
    """Collects usage per (step, provider, model). Calls made outside a step are attributed to 'other'."""

    def __init__(self) -> None:
        self._stage = "other"
        self._steps: dict[tuple[str, str, str], StepUsage] = {}

    @contextmanager
    def step(self, stage: str) -> Iterator[None]:
        previous, self._stage = self._stage, stage
        try:
            yield
        finally:
            self._stage = previous

    def record(
        self, provider: str, model: str, input_tokens: int, output_tokens: int, cost: float | None = None
    ) -> None:
        key = (self._stage, provider, model)
        usage = self._steps.setdefault(key, StepUsage(self._stage, provider, model))
        usage.calls += 1
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        if cost is not None:
            usage.reported_cost += cost
            usage.calls_with_cost += 1

    def set_items(self, stage: str, items: int) -> None:
        """How many jobs the step processed, so per-job averages ignore retries and batching."""
        for (step_stage, _, _), usage in self._steps.items():
            if step_stage == stage:
                usage.items = items

    def take(self, stage: str) -> list[StepUsage]:
        """Remove and return what was recorded for `stage`."""
        keys = [key for key in self._steps if key[0] == stage]
        return [self._steps.pop(key) for key in keys]


async def save_usage(session: AsyncSession, steps: list[StepUsage]) -> None:
    for usage in steps:
        if usage.calls == 0:
            continue
        session.add(
            LlmUsage(
                stage=usage.stage,
                provider=usage.provider,
                model=usage.model,
                calls=usage.calls,
                items=usage.items or usage.calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=round(usage.cost, 6),
                cost_estimated=usage.estimated,
            )
        )
    await session.commit()


def describe(steps: list[StepUsage]) -> str:
    """One line per model: '412 calls, 2.1M in / 380k out tokens, $0.42 (reported)'."""
    if not steps:
        return "no model calls"

    def short(n: int) -> str:
        return f"{n / 1_000_000:.2f}M" if n >= 1_000_000 else f"{n / 1000:.0f}k" if n >= 1000 else str(n)

    parts = []
    for u in steps:
        basis = "estimated" if u.estimated else "reported"
        parts.append(
            f"{u.model}: {u.calls} calls, {short(u.input_tokens)} in / {short(u.output_tokens)} out tokens, "
            f"${u.cost:.4f} ({basis})"
        )
    return "; ".join(parts)
