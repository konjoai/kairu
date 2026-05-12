"""Prompt Shield — rule-based content policy at the API boundary.

Screens prompts for injection attacks, jailbreak phrases, and PII before
tokenization. Zero new runtime dependencies (stdlib only). Fail-open design:
shield errors log a warning and default to ALLOWED to avoid blocking
legitimate users on bugs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import auto
from typing import Optional

# StrEnum was added in 3.11; shim for 3.10 compatibility.
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover — Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python 3.10."""


logger = logging.getLogger("kairu.shield")

_MAX_REASON_CHARS = 200


class ShieldVerdict(StrEnum):
    """Outcome of a shield check."""

    ALLOWED = auto()
    FLAGGED = auto()
    BLOCKED = auto()


@dataclass(frozen=True)
class ShieldResult:
    """Immutable result returned by :meth:`PromptShield.check`."""

    verdict: ShieldVerdict
    reason: str
    confidence: float
    matched_rule: Optional[str]


@dataclass
class ShieldRule:
    """A single named policy rule matching a compiled regex pattern."""

    name: str
    verdict: ShieldVerdict
    pattern: re.Pattern  # type: ignore[type-arg]
    confidence: float
    reason_template: str
    priority: int = 0


@dataclass
class ShieldConfig:
    """Configuration for :class:`PromptShield`."""

    rules: list[ShieldRule] = field(default_factory=list)
    pii_patterns: list[re.Pattern] = field(default_factory=list)  # type: ignore[type-arg]
    max_prompt_length: int = 10_000
    enable_default_rules: bool = True
    enable_pii_detection: bool = True
    block_on_pii: bool = False


def _default_rules() -> list[ShieldRule]:
    """Return the built-in rule set (≥ 10 rules).

    Rules cover prompt injection, jailbreak phrases, and roleplay bypass.
    All patterns are case-insensitive and compiled once at module load.
    """
    return [
        ShieldRule(
            name="injection_ignore_previous",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions?", re.I),
            confidence=0.95,
            reason_template="injection pattern detected: {match}",
            priority=10,
        ),
        ShieldRule(
            name="injection_disregard_all",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"disregard\s+all\s+(?:previous\s+)?(?:instructions?|rules?|constraints?)", re.I),
            confidence=0.95,
            reason_template="injection pattern detected: {match}",
            priority=10,
        ),
        ShieldRule(
            name="injection_forget_system_prompt",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"forget\s+(?:your\s+)?(?:system\s+prompt|instructions?|context|training)", re.I),
            confidence=0.90,
            reason_template="injection pattern detected: {match}",
            priority=10,
        ),
        ShieldRule(
            name="injection_no_restrictions",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions?", re.I),
            confidence=0.92,
            reason_template="injection pattern detected: {match}",
            priority=9,
        ),
        ShieldRule(
            name="jailbreak_dan_mode",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"\bDAN\s+mode\b", re.I),
            confidence=0.97,
            reason_template="jailbreak pattern detected: {match}",
            priority=10,
        ),
        ShieldRule(
            name="jailbreak_developer_mode",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"developer\s+mode\s+enabled", re.I),
            confidence=0.95,
            reason_template="jailbreak pattern detected: {match}",
            priority=10,
        ),
        ShieldRule(
            name="jailbreak_you_are_now",
            verdict=ShieldVerdict.FLAGGED,
            pattern=re.compile(r"you\s+are\s+now\s+(?:an?\s+)?(?:AI\s+)?(?:without|that\s+can|able\s+to)", re.I),
            confidence=0.75,
            reason_template="jailbreak pattern detected: {match}",
            priority=7,
        ),
        ShieldRule(
            name="jailbreak_pretend_no_ai_restrictions",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"pretend\s+you\s+are\s+an?\s+AI\s+without(?:\s+any)?\s+restrictions?", re.I),
            confidence=0.93,
            reason_template="jailbreak pattern detected: {match}",
            priority=9,
        ),
        ShieldRule(
            name="roleplay_no_restrictions",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"pretend\s+you\s+have\s+no\s+restrictions?", re.I),
            confidence=0.92,
            reason_template="roleplay bypass detected: {match}",
            priority=9,
        ),
        ShieldRule(
            name="roleplay_act_as_trained",
            verdict=ShieldVerdict.FLAGGED,
            pattern=re.compile(r"act\s+as\s+if\s+you\s+were\s+(?:not\s+)?trained", re.I),
            confidence=0.80,
            reason_template="roleplay bypass detected: {match}",
            priority=7,
        ),
        ShieldRule(
            name="injection_override_instructions",
            verdict=ShieldVerdict.BLOCKED,
            pattern=re.compile(r"override\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|rules?|safety)", re.I),
            confidence=0.90,
            reason_template="injection pattern detected: {match}",
            priority=9,
        ),
        ShieldRule(
            name="jailbreak_token_manipulation",
            verdict=ShieldVerdict.FLAGGED,
            pattern=re.compile(r"(?:system\s+)?prompt\s+(?:injection|override|leak|extraction)", re.I),
            confidence=0.85,
            reason_template="prompt manipulation attempt detected: {match}",
            priority=8,
        ),
    ]


def _default_pii_patterns() -> list[re.Pattern]:  # type: ignore[type-arg]
    """Return compiled PII detection patterns: email, SSN, credit card, phone."""
    return [
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", re.I),
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
        re.compile(r"\b(?:\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
    ]


def _truncate_reason(reason: str) -> str:
    """Clamp reason string to _MAX_REASON_CHARS characters."""
    if len(reason) <= _MAX_REASON_CHARS:
        return reason
    return reason[: _MAX_REASON_CHARS - 3] + "..."


def _scan_rules(prompt: str, rules: list[ShieldRule]) -> Optional[ShieldResult]:
    """Scan rules in descending priority order; return first match or None."""
    for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
        m = rule.pattern.search(prompt)
        if m:
            reason = _truncate_reason(rule.reason_template.format(match=m.group()))
            return ShieldResult(
                verdict=rule.verdict,
                reason=reason,
                confidence=rule.confidence,
                matched_rule=rule.name,
            )
    return None


_PII_PATTERN_NAMES = ["pii_email", "pii_ssn", "pii_credit_card", "pii_phone"]


def _scan_pii(
    prompt: str,
    patterns: list[re.Pattern],  # type: ignore[type-arg]
    block_on_pii: bool,
) -> Optional[ShieldResult]:
    """Scan PII patterns; return first match as FLAGGED or BLOCKED."""
    for pattern, name in zip(patterns, _PII_PATTERN_NAMES + ["pii_custom"] * len(patterns)):
        m = pattern.search(prompt)
        if m:
            verdict = ShieldVerdict.BLOCKED if block_on_pii else ShieldVerdict.FLAGGED
            confidence = 0.80
            reason = _truncate_reason(f"PII detected ({name}): {m.group()}")
            return ShieldResult(
                verdict=verdict,
                reason=reason,
                confidence=confidence,
                matched_rule=name,
            )
    return None


_ALLOWED_RESULT = ShieldResult(
    verdict=ShieldVerdict.ALLOWED,
    reason="no policy violations detected",
    confidence=1.0,
    matched_rule=None,
)


class PromptShield:
    """Rule-based content policy gate for LLM prompts.

    Checks proceed in order: length cap → custom rules → default rules →
    PII detection. First match wins. Fail-open: exceptions log a warning
    and return ALLOWED.
    """

    def __init__(self, config: Optional[ShieldConfig] = None) -> None:
        """Initialise with an optional :class:`ShieldConfig`.

        If *config* is ``None``, the default rule set is used.
        """
        self._cfg: ShieldConfig = config if config is not None else ShieldConfig()
        self._default_rules: list[ShieldRule] = (
            _default_rules() if self._cfg.enable_default_rules else []
        )
        self._pii_patterns: list[re.Pattern] = (  # type: ignore[type-arg]
            _default_pii_patterns() + self._cfg.pii_patterns
            if self._cfg.enable_pii_detection
            else []
        )

    def check(self, prompt: str) -> ShieldResult:
        """Screen *prompt* against all configured policy rules.

        Steps (in order, first match wins):

        1. Length check: over ``max_prompt_length`` → BLOCKED.
        2. Custom rules (priority descending): first match → its verdict.
        3. Default rules (if ``enable_default_rules``): same priority scan.
        4. PII detection (if ``enable_pii_detection``).
        5. No match → ALLOWED.
        """
        try:
            return self._check_inner(prompt)
        except Exception:  # noqa: BLE001 — fail-open, never crash the caller
            logger.warning("PromptShield.check raised unexpectedly; defaulting to ALLOWED", exc_info=True)
            return _ALLOWED_RESULT

    def _check_inner(self, prompt: str) -> ShieldResult:
        """Inner screening logic — may raise; caller converts to fail-open."""
        if len(prompt) > self._cfg.max_prompt_length:
            return ShieldResult(
                verdict=ShieldVerdict.BLOCKED,
                reason=_truncate_reason(
                    f"prompt length {len(prompt)} exceeds limit {self._cfg.max_prompt_length}"
                ),
                confidence=1.0,
                matched_rule="max_prompt_length",
            )

        custom_hit = _scan_rules(prompt, self._cfg.rules)
        if custom_hit is not None:
            return custom_hit

        default_hit = _scan_rules(prompt, self._default_rules)
        if default_hit is not None:
            return default_hit

        pii_hit = _scan_pii(prompt, self._pii_patterns, self._cfg.block_on_pii)
        if pii_hit is not None:
            return pii_hit

        return _ALLOWED_RESULT

    def is_allowed(self, prompt: str) -> bool:
        """Return ``True`` iff :meth:`check` yields :attr:`ShieldVerdict.ALLOWED`."""
        return self.check(prompt).verdict == ShieldVerdict.ALLOWED


_DEFAULT_SHIELD: Optional[PromptShield] = None


def get_default_shield() -> PromptShield:
    """Return the module-level singleton :class:`PromptShield` (default config)."""
    global _DEFAULT_SHIELD
    if _DEFAULT_SHIELD is None:
        _DEFAULT_SHIELD = PromptShield()
    return _DEFAULT_SHIELD


__all__ = [
    "ShieldVerdict",
    "ShieldResult",
    "ShieldRule",
    "ShieldConfig",
    "PromptShield",
    "get_default_shield",
]
