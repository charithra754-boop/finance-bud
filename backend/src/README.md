# FinPilot Backend - Clean Architecture

This directory contains the refactored backend following Clean Architecture principles.

## Directory Structure

```
backend/src/
├── domain/              # Pure business logic (NO external dependencies)
│   ├── models/          # Domain entities (FinancialGoal, Constraint, Plan)
│   ├── services/        # Domain services (business rule implementations)
│   └── value_objects/   # Immutable value objects (Money, Percentage, TimeHorizon)
│
├── application/         # Use cases and application logic
│   ├── use_cases/       # Application use cases (GeneratePlan, VerifyConstraints)
│   ├── ports/           # Interfaces for infrastructure (IRepository, IMessageBus)
│   └── dto/             # Data Transfer Objects for use cases
│
├── infrastructure/      # External concerns and implementations
│   ├── agents/          # Agent implementations
│   ├── persistence/     # Database, repositories
│   ├── messaging/       # Message bus, communication framework
│   └── apis/            # External API clients (yfinance, Alpha Vantage)
│
└── presentation/        # API layer
    └── api/
        ├── v1/          # Version 1 endpoints
        ├── v2/          # Version 2 endpoints (future)
        └── dependencies.py  # FastAPI dependencies

```

## Layer Dependency Rules

**CRITICAL**: Dependencies must only flow inward:

```
Presentation → Application → Domain ← Infrastructure
                                ↑
                                │
                            (implements)
```

### Domain Layer (Core)
- **NO external dependencies** (only Python stdlib)
- Contains pure business logic
- Defines entities, value objects, domain services
- Example: `FinancialGoal`, `RiskCalculationService`

### Application Layer
- **Depends on**: Domain only
- Contains use cases (application-specific business rules)
- Defines ports (interfaces) for infrastructure
- Example: `GenerateFinancialPlanUseCase`

### Infrastructure Layer
- **Depends on**: Application ports (interfaces)
- Implements technical details
- Agents, databases, external APIs
- Example: `PlanningAgent implements IPlanningAgent`

### Presentation Layer
- **Depends on**: Application use cases
- FastAPI endpoints, request/response models
- No business logic here
- Example: `POST /api/v1/planning/generate`

## Migration Status

| Component | Status | Location |
|-----------|--------|----------|
| Domain Models | 🔄 In Progress | `domain/models/` |
| Use Cases | 📅 Planned | `application/use_cases/` |
| Agents | 📅 Planned | `infrastructure/agents/` |
| API v1 | 📅 Planned | `presentation/api/v1/` |

## Development Guidelines

### When adding new code:

1. **Start with domain** - What are the business entities and rules?
2. **Define use case** - What does the application need to do?
3. **Create port (interface)** - What infrastructure do we need?
4. **Implement infrastructure** - How do we actually do it?
5. **Expose via API** - How does the user access it?

### Example: Adding "Tax Optimization" Feature

```python
# 1. Domain (backend/src/domain/models/tax_strategy.py)
class TaxStrategy:
    """Pure business logic for tax optimization"""
    strategy_type: str
    estimated_savings: Decimal

# 2. Use Case (backend/src/application/use_cases/optimize_taxes.py)
class OptimizeTaxesUseCase:
    def __init__(self, tax_calculator: ITaxCalculator):
        self.calculator = tax_calculator

    def execute(self, request: OptimizeTaxRequest) -> TaxStrategy:
        # Business logic orchestration
        ...

# 3. Port (backend/src/application/ports/tax_calculator.py)
class ITaxCalculator(ABC):
    @abstractmethod
    def calculate_optimal_strategy(...) -> TaxStrategy:
        pass

# 4. Infrastructure (backend/src/infrastructure/calculators/tax_calculator.py)
class TaxCalculator(ITaxCalculator):
    def calculate_optimal_strategy(...) -> TaxStrategy:
        # Actual implementation with external libraries
        ...

# 5. Presentation (backend/src/presentation/api/v1/tax.py)
@router.post("/optimize")
async def optimize_taxes(
    request: OptimizeTaxRequest,
    use_case: OptimizeTaxesUseCase = Depends()
):
    return use_case.execute(request)
```

## Testing Strategy

```
tests/
├── unit/
│   ├── domain/       # Test business logic in isolation
│   ├── application/  # Test use cases with mocked ports
│   └── infrastructure/ # Test implementations
├── integration/      # Test across layers
└── contract/         # Test API contracts
```

## Key Principles

✅ **DO**:
- Keep domain layer pure (no external dependencies)
- Use dependency injection
- Program to interfaces, not implementations
- Write tests for each layer independently

❌ **DON'T**:
- Import infrastructure in domain
- Put business logic in presentation
- Create circular dependencies
- Skip writing interfaces for infrastructure

## References

- [ADR 001: Clean Architecture Refactoring](../../docs/adr/001-clean-architecture-refactoring.md)
- [Clean Architecture (Uncle Bob)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

**Migration Phase:** Phase 1 - Foundation & Safety Net
**Last Updated:** 2025-11-18
