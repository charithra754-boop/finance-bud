#!/usr/bin/env python3
"""
Check which data models are being used across the codebase
"""

import os
import re
from collections import defaultdict

# All data models from schemas.py
ALL_MODELS = [
    "AgentMessage", "AgentType", "AuditTrail", "ComplianceLevel", "ComplianceStatus",
    "Constraint", "ConstraintPriority", "ConstraintType", "DecisionPoint",
    "EnhancedPlanRequest", "ExecutionLog", "ExecutionStatus", "FinancialState",
    "MarketData", "MarketEventType", "MessageType", "PerformanceMetrics",
    "PlanStep", "Priority", "ReasoningTrace", "RegulatoryRequirement",
    "RiskLevel", "RiskProfile", "SearchPath", "SeverityLevel", "TaxContext",
    "TriggerEvent", "VerificationReport", "VerificationStatus"
]

def find_model_usage(directory="."):
    """Find where each model is used"""
    usage = defaultdict(list)

    # Search in Python files
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        if any(skip in root for skip in ['.git', 'node_modules', '__pycache__', 'build', 'dist', 'venv']):
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        for model in ALL_MODELS:
                            # Look for imports and usage
                            patterns = [
                                rf'\bfrom\s+data_models\.schemas\s+import\s+.*\b{model}\b',
                                rf'\b{model}\s*\(',  # Instantiation
                                rf':\s*{model}\b',   # Type hints
                                rf'->\s*{model}\b',  # Return type
                            ]

                            for pattern in patterns:
                                if re.search(pattern, content):
                                    usage[model].append(filepath)
                                    break

                except Exception as e:
                    pass

    return usage

def main():
    print("=" * 80)
    print("DATA MODEL USAGE ANALYSIS")
    print("=" * 80)

    usage = find_model_usage()

    used_models = set(usage.keys())
    unused_models = set(ALL_MODELS) - used_models

    print(f"\n✅ MODELS IN USE: {len(used_models)}/{len(ALL_MODELS)}")
    print("-" * 80)

    for model in sorted(used_models):
        files = usage[model]
        print(f"\n{model}:")
        print(f"  Used in {len(files)} files:")
        for f in files[:5]:  # Show first 5 files
            print(f"    - {f}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")

    if unused_models:
        print(f"\n\n⚠️  UNUSED MODELS: {len(unused_models)}")
        print("-" * 80)
        for model in sorted(unused_models):
            print(f"  - {model}")

        print("\n💡 RECOMMENDATION:")
        print("These models are defined but not currently used. Consider:")
        print("  1. Adding features that use these models")
        print("  2. Removing unused models to reduce complexity")
        print("  3. Keeping them for future features (if planned)")
    else:
        print("\n\n✅ ALL MODELS ARE BEING USED!")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(used_models)}/{len(ALL_MODELS)} models in use")
    print("=" * 80)

if __name__ == "__main__":
    main()
