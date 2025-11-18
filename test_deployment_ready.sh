#!/bin/bash
# Pre-Deployment Verification Script
# Tests all critical components before deployment

set -e  # Exit on error

echo "=================================="
echo "FinPilot MVP - Deployment Readiness Test"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

check_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((pass_count++))
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((fail_count++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

echo "1. Checking Python Backend..."
echo "------------------------------"

# Test Python import
if python3 -c "from main import app; print('OK')" 2>/dev/null; then
    check_pass "Backend imports successfully"
else
    check_fail "Backend import failed"
fi

# Test data models
if python3 -c "from data_models.schemas import *; print('OK')" 2>/dev/null; then
    check_pass "Data models import successfully"
else
    check_fail "Data models import failed"
fi

echo ""
echo "2. Checking Frontend Build..."
echo "------------------------------"

# Check if build directory exists
if [ -d "build" ] && [ -f "build/index.html" ]; then
    check_pass "Frontend build exists"
else
    check_warn "Frontend build not found (run 'npm run build')"
fi

# Check node_modules
if [ -d "node_modules" ]; then
    check_pass "Node modules installed"
else
    check_fail "Node modules not installed (run 'npm install')"
fi

echo ""
echo "3. Checking Configuration Files..."
echo "------------------------------"

# Check deployment configs
configs=("Procfile" "render.yaml" "railway.json" "fly.toml" "requirements-minimal.txt")
for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        check_pass "$config exists"
    else
        check_fail "$config missing"
    fi
done

echo ""
echo "4. Checking Documentation..."
echo "------------------------------"

docs=("DEPLOY.md" "MVP_READY.md" "QUICKSTART.md" "PRE_DEPLOYMENT_CHECKLIST.md")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        check_pass "$doc exists"
    else
        check_fail "$doc missing"
    fi
done

echo ""
echo "5. Checking Git Status..."
echo "------------------------------"

if git rev-parse --git-dir > /dev/null 2>&1; then
    check_pass "Git repository initialized"

    # Check if there are uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        check_warn "Uncommitted changes exist (commit before deploy)"
    else
        check_pass "No uncommitted changes"
    fi
else
    check_fail "Not a git repository"
fi

echo ""
echo "6. Checking Model Usage..."
echo "------------------------------"

if python3 check_model_usage.py > /dev/null 2>&1; then
    model_count=$(python3 -c "from data_models.schemas import *; import inspect; import data_models.schemas as s; print(len([n for n, o in inspect.getmembers(s) if inspect.isclass(o) and o.__module__ == 'data_models.schemas']))")
    check_pass "All $model_count data models verified"
else
    check_fail "Model usage check failed"
fi

echo ""
echo "7. Running Data Model Tests..."
echo "------------------------------"

if pytest data_models/test_schemas.py -q > /dev/null 2>&1; then
    check_pass "Data model tests passing (12/12)"
else
    check_fail "Data model tests failing"
fi

echo ""
echo "=================================="
echo "DEPLOYMENT READINESS SUMMARY"
echo "=================================="
echo ""
echo -e "Passed: ${GREEN}$pass_count${NC}"
echo -e "Failed: ${RED}$fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✅ READY FOR DEPLOYMENT!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. git add ."
    echo "  2. git commit -m 'Ready for deployment'"
    echo "  3. git push origin main"
    echo "  4. Deploy using DEPLOY.md guide"
    echo ""
    exit 0
else
    echo -e "${RED}❌ NOT READY FOR DEPLOYMENT${NC}"
    echo ""
    echo "Fix the failed checks above before deploying."
    echo ""
    exit 1
fi
