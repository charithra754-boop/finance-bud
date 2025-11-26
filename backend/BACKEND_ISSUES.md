# Backend Known Issues

**Last Updated**: 2025-11-26

## Critical Issues (Blocking Server Startup)

### 1. Missing `get_logger` Function in `utils/logger.py`

**Status**: 🔴 Critical
**Impact**: Information Retrieval Agent fails to initialize
**File**: `agents/retriever.py:31`

**Error**:
```
ImportError: cannot import name 'get_logger' from 'utils.logger'
```

**Details**:
- The `InformationRetrievalAgent` in `agents/retriever.py` tries to import `get_logger` from `utils.logger`
- This function doesn't exist in the logger module
- When the import fails, the system tries to fall back to a mock agent

**Fix Required**:
- Add `get_logger()` function to `utils/logger.py`, or
- Update `agents/retriever.py` to use the correct logger import

---

### 2. Missing Mock Base Agent Module

**Status**: 🔴 Critical
**Impact**: All mock agent fallbacks fail
**File**: `tests/mocks/mock_interfaces.py:18`

**Error**:
```
ModuleNotFoundError: No module named 'tests.mocks.base_agent'
```

**Details**:
- When real agents fail to import, the factory tries to use mock agents as fallbacks
- `tests/mocks/mock_interfaces.py` tries to import from `.base_agent`
- This module doesn't exist in `tests/mocks/`

**Fix Required**:
- Create `tests/mocks/base_agent.py` with a mock BaseAgent class, or
- Update the import path in `mock_interfaces.py` to use the real `agents.base_agent`, or
- Remove the mock fallback mechanism and fix the real agents

---

### 3. Missing API Documentation Files

**Status**: 🟡 Non-blocking (worked around)
**Impact**: API package imports failed initially
**File**: `api/__init__.py`

**Error** (Fixed):
```
ModuleNotFoundError: No module named 'api.documentation'
ModuleNotFoundError: No module named 'api.endpoints'
```

**Workaround Applied**:
- Commented out imports in `api/__init__.py` for:
  - `api.documentation.APIDocumentationStandards`
  - `api.endpoints.AgentEndpoints`
  - `api.endpoints.endpoint_registry`

**Fix Required** (Optional):
- Create these missing modules if they're needed, or
- Remove the commented imports if they're not part of the architecture

---

## Dependency Issues (Resolved)

### ✅ Missing Python Packages

**Status**: ✅ Resolved
**Packages Installed**:
- `pydantic-settings` - Required for config management
- `numpy` - Required for planning agent
- `pandas` - Data processing
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning

**Note**: Some packages from `requirements.txt` may still be missing. Run `make install` for full installation.

---

## Agent Initialization Status

| Agent | Status | Notes |
|-------|--------|-------|
| Orchestration Agent (OA) | ✅ Working | Initializes successfully |
| Planning Agent (PA) | ✅ Working | Initializes successfully after numpy install |
| Information Retrieval Agent (IRA) | ❌ Failed | Missing `get_logger` import |
| Verification Agent (VA) | ⚠️ Unknown | Not reached due to IRA failure |
| Execution Agent (EA) | ⚠️ Unknown | Not reached due to IRA failure |
| Conversational Agent (CA) | ⚠️ Degraded | Works but Ollama unavailable (using fallback) |

---

## Optional Enhancements

### Missing Ollama Integration

**Status**: 🟡 Non-critical (has fallback)
**Impact**: Conversational agent uses rule-based processing instead of LLM

**Error**:
```
WARNING - Ollama not available - falling back to rule-based processing
Import error: No module named 'ollama'
```

**Fix**:
```bash
pip install ollama
```

---

## Recommended Fix Priority

1. **High Priority** (Required for server startup):
   - [ ] Fix `get_logger` import in `utils/logger.py` or `agents/retriever.py`
   - [ ] Fix or remove mock agent fallback mechanism

2. **Medium Priority** (Improves functionality):
   - [ ] Install remaining dependencies: `make install`
   - [ ] Create missing API documentation modules (if needed)
   - [ ] Install Ollama for LLM support: `pip install ollama`

3. **Low Priority** (Code cleanup):
   - [ ] Review and clean up commented imports in `api/__init__.py`
   - [ ] Verify all agents can initialize
   - [ ] Add missing test mock modules if mock system is needed

---

## Quick Fix Commands

```bash
# Check what dependencies are missing
make check-deps

# Install all dependencies
make install

# Clean up and fresh start
make clean
make run

# Debug imports
make debug-imports
```

---

## Notes for Developers

- The Makefile infrastructure is working correctly ✅
- Frontend is fully functional on port 3000 ✅
- Backend code structure is incomplete or has import issues
- Consider whether mock agent fallback system is needed
- May want to simplify agent initialization to fail fast rather than fall back to mocks

---

## How to Test Fixes

After making fixes, test with:

```bash
# Clean environment
make clean

# Try starting server
make run

# Or debug imports first
make debug-imports
```

Check the logs for:
- Agent initialization messages
- Import errors
- Successful startup message: "Application startup complete"
