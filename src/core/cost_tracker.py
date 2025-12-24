"""
Cost tracking infrastructure for API calls.

Wraps every API call to record model, tokens, cost, and metadata.
Provides per-question breakdowns and flags expensive patterns.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from .models import (
    APICallLog,
    CostSummary,
    CallType,
    ModelType,
)


# Pricing per 1M tokens (as of late 2024/early 2025)
PRICING = {
    ModelType.OPUS: {
        "input": 15.00,   # $15 per 1M input tokens
        "output": 75.00,  # $75 per 1M output tokens
    },
    ModelType.FLASH: {
        "input": 0.10,    # $0.10 per 1M input tokens
        "output": 0.40,   # $0.40 per 1M output tokens
    },
}


class CostTracker:
    """
    Tracks costs for all API calls during a question session.

    Features:
    - Per-call logging with full metadata
    - Aggregated cost summaries
    - Warnings for expensive patterns
    - JSON export for analysis
    """

    def __init__(
        self,
        question_id: str,
        log_dir: str = "logs",
        soft_budget: Optional[float] = None,
        hard_budget: Optional[float] = None,
    ):
        self.question_id = question_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.soft_budget = soft_budget or float(os.getenv("SOFT_BUDGET_LIMIT", "1.0"))
        self.hard_budget = hard_budget or float(os.getenv("HARD_BUDGET_LIMIT", "5.0"))

        self.calls: list[APICallLog] = []
        self.disambiguation_iterations = 0
        self.max_verification_depth = 0
        self.warnings: list[str] = []

    def calculate_cost(
        self,
        model: ModelType,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost in USD for a given token count."""
        pricing = PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def log_call(
        self,
        model: ModelType,
        call_type: CallType,
        input_tokens: int,
        output_tokens: int,
        prompt_preview: str = "",
        response_preview: str = "",
        duration_ms: int = 0,
    ) -> APICallLog:
        """Log an API call and check budget limits."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        log_entry = APICallLog(
            model=model,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            prompt_preview=prompt_preview[:200] if prompt_preview else "",
            response_preview=response_preview[:200] if response_preview else "",
            duration_ms=duration_ms,
        )

        self.calls.append(log_entry)
        self._check_budgets_and_patterns()

        return log_entry

    def _check_budgets_and_patterns(self):
        """Check for budget limits and expensive patterns."""
        total = self.get_total_cost()

        # Budget warnings
        if total >= self.soft_budget and f"Soft budget (${self.soft_budget}) exceeded" not in self.warnings:
            self.warnings.append(f"Soft budget (${self.soft_budget}) exceeded at ${total:.4f}")

        if total >= self.hard_budget:
            self.warnings.append(f"HARD BUDGET (${self.hard_budget}) EXCEEDED at ${total:.4f}")
            raise BudgetExceededError(f"Hard budget limit of ${self.hard_budget} exceeded")

        # Pattern warnings
        if self.disambiguation_iterations >= 5:
            warning = f"Disambiguation loop running {self.disambiguation_iterations}+ times"
            if warning not in self.warnings:
                self.warnings.append(warning)

        if self.max_verification_depth >= 3:
            warning = f"Deep verification chain: depth {self.max_verification_depth}"
            if warning not in self.warnings:
                self.warnings.append(warning)

    def increment_disambiguation(self):
        """Track disambiguation iteration count."""
        self.disambiguation_iterations += 1
        self._check_budgets_and_patterns()

    def update_verification_depth(self, depth: int):
        """Track max verification depth."""
        if depth > self.max_verification_depth:
            self.max_verification_depth = depth
            self._check_budgets_and_patterns()

    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return sum(call.cost_usd for call in self.calls)

    def get_summary(self) -> CostSummary:
        """Generate a complete cost summary."""
        opus_calls = [c for c in self.calls if c.model == ModelType.OPUS]
        flash_calls = [c for c in self.calls if c.model == ModelType.FLASH]

        return CostSummary(
            question_id=self.question_id,
            total_cost_usd=self.get_total_cost(),
            total_calls=len(self.calls),
            opus_cost_usd=sum(c.cost_usd for c in opus_calls),
            opus_calls=len(opus_calls),
            flash_cost_usd=sum(c.cost_usd for c in flash_calls),
            flash_calls=len(flash_calls),
            disambiguation_iterations=self.disambiguation_iterations,
            max_verification_depth=self.max_verification_depth,
            warnings=self.warnings.copy(),
            calls=self.calls.copy(),
        )

    def save_log(self) -> Path:
        """Save the complete log to a JSON file."""
        summary = self.get_summary()
        filename = f"{self.question_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename

        with open(filepath, "w") as f:
            json.dump(summary.model_dump(mode="json"), f, indent=2, default=str)

        return filepath

    def get_breakdown_string(self) -> str:
        """Get a human-readable cost breakdown."""
        summary = self.get_summary()
        lines = [
            f"=== Cost Breakdown for Question {self.question_id[:8]}... ===",
            f"Total Cost: ${summary.total_cost_usd:.4f} across {summary.total_calls} calls",
            f"",
            f"By Model:",
            f"  Opus:  ${summary.opus_cost_usd:.4f} ({summary.opus_calls} calls)",
            f"  Flash: ${summary.flash_cost_usd:.4f} ({summary.flash_calls} calls)",
            f"",
            f"Iterations:",
            f"  Disambiguation: {summary.disambiguation_iterations}",
            f"  Max Verification Depth: {summary.max_verification_depth}",
        ]

        if summary.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in summary.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)


class BudgetExceededError(Exception):
    """Raised when the hard budget limit is exceeded."""
    pass
