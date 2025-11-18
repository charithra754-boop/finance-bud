# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for FinPilot VP-MAS.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences.

## ADR Format

Each ADR follows this structure:

```markdown
# ADR NNN: Title

**Status:** [Proposed | Accepted | Deprecated | Superseded]
**Date:** YYYY-MM-DD
**Deciders:** Who made the decision
**Context:** Which phase/feature

## Context and Problem Statement
[Describe the context and problem]

## Decision Drivers
[Key factors influencing the decision]

## Considered Options
[List of options considered]

## Decision Outcome
[The chosen solution and rationale]

## Consequences
[Positive and negative outcomes]

## Compliance
[Alignment with principles/standards]

## References
[Links to resources]
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-clean-architecture-refactoring.md) | Clean Architecture Refactoring | Accepted | 2025-11-18 |

## How to Create a New ADR

1. Copy the template above
2. Use the next number in sequence
3. Use kebab-case for filename: `NNN-descriptive-title.md`
4. Fill in all sections
5. Get review and approval from team
6. Update this README with the new ADR

## ADR Statuses

- **Proposed**: Under discussion
- **Accepted**: Decision made and being implemented
- **Deprecated**: No longer applicable
- **Superseded**: Replaced by a newer ADR

## When to Write an ADR

Write an ADR for decisions that:
- Affect the overall architecture
- Have significant impact on multiple components
- Are difficult or costly to reverse
- Set precedents for future development
- Involve trade-offs between competing concerns

## Examples of ADR-Worthy Decisions

- Choice of architectural pattern (Clean Architecture, CQRS, etc.)
- Major technology selections (frameworks, databases, message queues)
- API versioning strategy
- Security and authentication approach
- Data model design changes
- Deployment and infrastructure decisions
- Testing strategy

## Not ADR-Worthy

- Minor bug fixes
- Code style preferences (use linting config instead)
- Implementation details that don't affect architecture
- Temporary workarounds

---

*For more information on ADRs, see: [Architecture Decision Records](https://adr.github.io/)*
