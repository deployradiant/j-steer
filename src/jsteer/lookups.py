from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .schemas import CustomConceptInput, InjectionTarget, JSpaceInjection


@dataclass(frozen=True)
class LookupEntry:
    id: str
    aliases: tuple[str, ...]
    title: str
    source: str
    summary: str
    injected_text: str


LOOKUP_ENTRIES: tuple[LookupEntry, ...] = (
    LookupEntry(
        id="security-controls",
        aliases=("soc 2", "soc2", "audit log", "audit logs", "sso", "single sign-on"),
        title="Security controls",
        source="HeliosCRM canon/security-controls",
        summary=(
            "HeliosCRM positioning should cite SOC 2 posture, admin-visible audit logs, SSO, "
            "and governed connector access without inventing certification details."
        ),
        injected_text=(
            "Lookup injection: HeliosCRM security context includes SOC 2 posture, audit logs, SSO, "
            "admin controls, and governed connector access. Keep claims bounded to provided evidence."
        ),
    ),
    LookupEntry(
        id="customer-data-boundary",
        aliases=(
            "private customer record",
            "private customer records",
            "customer data",
            "confidential",
            "pii",
            "ssn",
        ),
        title="Customer data boundary",
        source="HeliosCRM canon/data-handling",
        summary=(
            "Customer records should be treated as confidential; responses should emphasize scoped access, "
            "auditability, and explicit administrator approval."
        ),
        injected_text=(
            "Lookup injection: private customer records require scoped access, auditability, least privilege, "
            "and explicit admin approval before operational use."
        ),
    ),
    LookupEntry(
        id="buying-workflow",
        aliases=("pricing", "budget", "approval", "procurement", "buyer", "buying"),
        title="Buying workflow",
        source="HeliosCRM canon/buyer-workflow",
        summary=(
            "Enterprise buyers need procurement-ready language: pricing discussion, budget owner, approval path, "
            "security review, and implementation handoff."
        ),
        injected_text=(
            "Lookup injection: enterprise buying context should mention pricing discussion, budget ownership, "
            "approval workflow, procurement review, security review, and rollout handoff."
        ),
    ),
    LookupEntry(
        id="competitor-boundary",
        aliases=("salesforce", "hubspot", "zendesk", "competitor", "competitors", "alternative", "alternatives"),
        title="Competitor boundary",
        source="HeliosCRM canon/competitive-claims",
        summary=(
            "Competitor mentions should not become market-leader claims; keep comparisons factual, neutral, "
            "and anchored to HeliosCRM governed workflow strengths."
        ),
        injected_text=(
            "Lookup injection: competitor context should redirect to HeliosCRM governed workflow automation, "
            "secure connectors, audit trails, and verified evidence instead of unsupported superiority claims."
        ),
    ),
)


def lookup_jspace_injections(
    prompt: str,
    *,
    target: InjectionTarget,
    custom_concepts: Sequence[CustomConceptInput] = (),
    default_targets: Sequence[str] = (),
    limit: int = 4,
) -> list[JSpaceInjection]:
    """Return deterministic keyword lookups that can be injected into concept/J-space paths."""

    text = prompt.casefold()
    injections: list[JSpaceInjection] = []
    for entry in LOOKUP_ENTRIES:
        matched = [alias for alias in entry.aliases if _contains_alias(text, alias)]
        if not matched:
            continue
        concept_targets = _concept_targets_for_entry(entry, matched, text, custom_concepts, default_targets)
        injections.append(
            JSpaceInjection(
                id=f"{target}:{entry.id}",
                keyword=matched[0],
                title=entry.title,
                source=entry.source,
                summary=entry.summary,
                injected_text=entry.injected_text,
                concept_targets=concept_targets,
                target=target,
            )
        )
        if len(injections) >= limit:
            break
    return injections


def _concept_targets_for_entry(
    entry: LookupEntry,
    matched_aliases: Sequence[str],
    prompt_text: str,
    custom_concepts: Sequence[CustomConceptInput],
    default_targets: Sequence[str],
) -> list[str]:
    targets = list(dict.fromkeys(target for target in default_targets if target))
    for concept in custom_concepts:
        if _custom_concept_matches_entry(concept, entry, matched_aliases, prompt_text):
            targets.append(concept.label.strip())
    return list(dict.fromkeys(target for target in targets if target))


def _custom_concept_matches_entry(
    concept: CustomConceptInput,
    entry: LookupEntry,
    matched_aliases: Sequence[str],
    prompt_text: str,
) -> bool:
    entry_aliases = tuple(alias.casefold() for alias in entry.aliases)
    matched = tuple(alias.casefold() for alias in matched_aliases)
    for keyword in concept.keywords:
        value = keyword.strip().casefold()
        if not value or not _contains_alias(prompt_text, value):
            continue
        if value in entry_aliases or any(value in alias or alias in value for alias in matched):
            return True
    return False


def _contains_alias(text: str, alias: str) -> bool:
    value = alias.strip().casefold()
    if not value:
        return False
    if value.isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", text) is not None
    return value in text
