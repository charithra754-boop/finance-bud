# ADR 001: Clean Architecture Refactoring

**Status:** Accepted
**Date:** 2025-11-18
**Deciders:** Architecture Team
**Context:** Phase 1 - Foundation & Safety Net

## Context and Problem Statement

The FinPilot VP-MAS codebase has accumulated significant technical debt:

- **God files**: `planning_agent.py` (4,112 lines), `execution_endpoints.py` (30,708 lines), `advanced_planning_capabilities.py` (3,014 lines)
- **Tight coupling**: Agents directly instantiate and call each other
- **Mixed concerns**: Business logic, infrastructure, and presentation layers intermingled
- **Poor testability**: Difficult to unit test due to concrete dependencies
- **Scalability issues**: Monolithic structure prevents independent agent deployment
- **Development velocity**: New features require touching many files; difficult onboarding

This technical debt is impacting:
1. Development velocity (slow feature development)
2. Code comprehension (hard for new developers)
3. Performance/scaling (memory leaks, tight coupling prevents horizontal scaling)

## Decision Drivers

- **Conservative approach required**: Production system, must maintain stability
- **Timeline**: 10-12 weeks for comprehensive refactoring
- **API compatibility**: Minor breaking changes acceptable with versioning
- **Key pain points**: Development velocity, performance/scaling, code comprehension

## Considered Options

### Option 1: Big Bang Rewrite
**Pros:**
- Clean slate, perfect architecture
- No legacy constraints

**Cons:**
- High risk, long development freeze
- Difficult to maintain existing system during rewrite
- All-or-nothing deployment
- **REJECTED** - Too risky for conservative approach

### Option 2: Incremental Refactoring (No Architecture Change)
**Pros:**
- Low risk, gradual changes
- Continuous delivery

**Cons:**
- Doesn't address root architectural issues
- Technical debt persists
- Limited long-term benefit
- **REJECTED** - Doesn't solve core problems

### Option 3: Clean Architecture with Strangler Fig Pattern (SELECTED)
**Pros:**
- Addresses root causes (layering, coupling, testability)
- Gradual migration reduces risk
- Can validate each step before proceeding
- Industry-proven patterns (Clean Architecture, DDD, SOLID)
- Enables future scalability

**Cons:**
- Requires discipline and consistency
- Temporary complexity during migration
- 10-12 week timeline

## Decision Outcome

**Chosen option:** Clean Architecture with Strangler Fig Pattern

We will refactor FinPilot to follow Clean Architecture principles over 6 phases (10-12 weeks):

### Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│  (FastAPI endpoints, API contracts, request/response)   │
│             backend/src/presentation/api/                │
└───────────────────┬─────────────────────────────────────┘
                    │ depends on ↓
┌─────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                      │
│     (Use cases, orchestration, application services)    │
│            backend/src/application/use_cases/            │
│            backend/src/application/ports/ (interfaces)   │
└───────────────────┬─────────────────────────────────────┘
                    │ depends on ↓
┌─────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                         │
│    (Pure business logic, entities, value objects)       │
│         backend/src/domain/models/                       │
│         backend/src/domain/services/                     │
│         ⚠️  ZERO EXTERNAL DEPENDENCIES                   │
└──────────────────────────────────────────────┬──────────┘
                                               ↑ implements
┌─────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                    │
│   (Agent implementations, external APIs, databases)     │
│         backend/src/infrastructure/agents/               │
│         backend/src/infrastructure/persistence/          │
│         backend/src/infrastructure/messaging/            │
└─────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Dependency Rule**: Dependencies point inward only
   - Infrastructure → Application → Domain
   - Domain has NO external dependencies

2. **Separation of Concerns**
   - **Domain**: Business rules, entities, core logic
   - **Application**: Use cases, workflows, orchestration
   - **Infrastructure**: Technical implementations (agents, APIs, DB)
   - **Presentation**: API layer, request/response handling

3. **Dependency Injection**
   - All dependencies injected via constructors
   - Use interfaces (ports) for infrastructure
   - No service locator pattern

4. **Interface Segregation**
   - Define clear contracts (`IAgent`, `IPlanningAgent`, `IMessageBus`)
   - Agents communicate via message bus, not direct calls

5. **Single Responsibility**
   - Each class/module has one reason to change
   - Break up god files into focused modules

### Migration Strategy: Strangler Fig Pattern

1. **Create new structure alongside old** (no big bang)
2. **Implement new code in new structure** (clean architecture)
3. **Route to new implementation via feature flags**
4. **Validate new implementation** (tests, monitoring)
5. **Remove old code once validated** (gradual cleanup)

Example:
```python
# Old (current)
class PlanningAgent:
    def generate_plan(...):
        # 4,000 lines of code here

# New (parallel implementation)
class PlanningAgentV2:
    def __init__(self, search_module: IGuidedSearch,
                 adjustment: IPlanAdjustment):
        # Dependencies injected, delegated to focused modules

# Feature flag routing
if feature_flags.use_new_planning_agent:
    return PlanningAgentV2(...)
else:
    return PlanningAgent(...)  # Old implementation
```

### Phase Breakdown

**Phase 1 (Weeks 1-2): Foundation & Safety Net** ✅ In Progress
- Comprehensive integration tests
- Contract tests for AgentMessage
- API contract documentation
- Performance baseline
- **This ADR**
- Create target directory structure
- Define core interfaces

**Phase 2 (Weeks 3-4): Data Model Reorganization**
- Split `schemas.py` into domain/application/infrastructure
- Create value objects
- Implement adapters for backward compatibility

**Phase 3 (Weeks 5-7): God File Decomposition**
- Break up `planning_agent.py` (4,112 lines)
- Break up `advanced_planning_capabilities.py` (3,014 lines)
- Break up `conversational_agent.py` (2,476 lines)
- Consolidate API endpoints (remove 30K line duplicate app)

**Phase 4 (Weeks 8-9): Clean Architecture Implementation**
- Create domain services
- Define application use cases
- Implement repository pattern
- Full dependency injection

**Phase 5 (Week 10): Testing & Performance**
- Reorganize test suite
- Fix memory leaks (unbounded correlation tracker)
- Performance optimization
- Validate baselines

**Phase 6 (Weeks 11-12): Frontend & Polish**
- Standardize React structure
- Generate TypeScript types from OpenAPI
- Documentation updates
- Final cleanup

## Consequences

### Positive

✅ **Better separation of concerns** - Clear boundaries between layers
✅ **Improved testability** - Can unit test domain logic without infrastructure
✅ **Enhanced maintainability** - Smaller, focused files easier to understand
✅ **Scalability** - Can deploy agents independently via message bus
✅ **Development velocity** - New features touch fewer files
✅ **Onboarding** - Clear architecture makes codebase navigable
✅ **Performance** - Can optimize hot paths, fix memory leaks

### Negative

⚠️ **Temporary complexity** - Dual implementations during migration
⚠️ **Learning curve** - Team must understand Clean Architecture
⚠️ **Discipline required** - Must consistently follow patterns
⚠️ **Migration effort** - 10-12 weeks of focused refactoring

### Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Performance regression | Establish baseline, monitor after each phase |
| Breaking changes | Contract tests, API versioning, feature flags |
| Incomplete migration | Clear phase gates, don't proceed if criteria not met |
| Scope creep | Stick to 6 phases, defer nice-to-haves |
| Team resistance | Documentation, pair programming, gradual rollout |

## Compliance

This ADR complies with:
- **SOLID principles**: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **Clean Architecture**: Dependency rule, layer separation
- **Domain-Driven Design**: Ubiquitous language, bounded contexts
- **12-Factor App**: Configuration, dependencies, processes

## Validation Criteria

Each phase must meet these criteria before proceeding:

1. **All tests pass** (unit, integration, contract)
2. **Performance within 10% of baseline** (no major regression)
3. **Code coverage ≥ 80%** on critical paths
4. **Zero critical security issues** (bandit, safety scans)
5. **API contracts maintained** (no unversioned breaking changes)
6. **Documentation updated** (README, architecture diagrams)

## References

- [Clean Architecture (Robert C. Martin)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design (Eric Evans)](https://www.domainlanguage.com/ddd/)
- [Strangler Fig Pattern (Martin Fowler)](https://martinfowler.com/bliki/StranglerFigApplication.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)

## Notes

- This is a **living document** - will be updated as we learn
- Future ADRs will detail specific implementation decisions
- Expected follow-up ADRs:
  - 002: Data Model Versioning Strategy
  - 003: Message Bus Implementation
  - 004: Repository Pattern for Financial State
  - 005: API Versioning and Migration Path
  - 006: Testing Strategy and Coverage Goals

---

**Last Updated:** 2025-11-18
**Next Review:** After Phase 2 completion
