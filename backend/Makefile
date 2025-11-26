.PHONY: help install install-dev run test test-unit test-integration clean lint format check-deps

# Default Python interpreter
PYTHON := python3
PIP := pip3

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)FinPilot Backend - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies (same as install for now)
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(PIP) install -r requirements.txt

install-quick: ## Install only critical dependencies for quick start
	@echo "$(GREEN)Installing critical dependencies...$(NC)"
	$(PIP) install fastapi uvicorn pydantic pydantic-settings numpy pandas scipy scikit-learn networkx

run: ## Start the FastAPI development server
	@echo "$(GREEN)Starting FinPilot backend server...$(NC)"
	@echo "$(YELLOW)Server will be available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API docs at: http://localhost:8000/docs$(NC)"
	PYTHONPATH=. uvicorn main:app --reload --port 8000

run-prod: ## Start the server in production mode (no reload)
	@echo "$(GREEN)Starting FinPilot backend server (production mode)...$(NC)"
	PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port 8000

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	PYTHONPATH=. $(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	PYTHONPATH=. $(PYTHON) -m pytest tests/integration/ -v

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	PYTHONPATH=. $(PYTHON) -m pytest tests/performance/ -v

lint: ## Run code linters
	@echo "$(GREEN)Running linters...$(NC)"
	-$(PYTHON) -m pylint agents/ api/ utils/ || true
	-$(PYTHON) -m flake8 agents/ api/ utils/ || true

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	-$(PYTHON) -m black agents/ api/ utils/ tests/ main.py config.py || true
	-$(PYTHON) -m isort agents/ api/ utils/ tests/ main.py config.py || true

check-deps: ## Check if critical dependencies are installed
	@echo "$(GREEN)Checking critical dependencies...$(NC)"
	@$(PYTHON) -c "import fastapi" 2>/dev/null && echo "  ✓ fastapi" || echo "  ✗ fastapi $(RED)(missing)$(NC)"
	@$(PYTHON) -c "import uvicorn" 2>/dev/null && echo "  ✓ uvicorn" || echo "  ✗ uvicorn $(RED)(missing)$(NC)"
	@$(PYTHON) -c "import pydantic" 2>/dev/null && echo "  ✓ pydantic" || echo "  ✗ pydantic $(RED)(missing)$(NC)"
	@$(PYTHON) -c "import pydantic_settings" 2>/dev/null && echo "  ✓ pydantic-settings" || echo "  ✗ pydantic-settings $(RED)(missing)$(NC)"
	@$(PYTHON) -c "import numpy" 2>/dev/null && echo "  ✓ numpy" || echo "  ✗ numpy $(RED)(missing)$(NC)"
	@$(PYTHON) -c "import pandas" 2>/dev/null && echo "  ✓ pandas" || echo "  ✗ pandas $(RED)(missing)$(NC)"
	@$(PYTHON) -c "import networkx" 2>/dev/null && echo "  ✓ networkx" || echo "  ✗ networkx $(RED)(missing)$(NC)"

clean: ## Clean up cache and temporary files
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

shell: ## Open Python shell with PYTHONPATH set
	@echo "$(GREEN)Opening Python shell...$(NC)"
	PYTHONPATH=. $(PYTHON)

api-docs: ## Open API documentation in browser
	@echo "$(GREEN)API docs should be available at:$(NC)"
	@echo "  http://localhost:8000/docs"
	@echo "  http://localhost:8000/redoc"

dev: install-quick run ## Quick setup and run (install critical deps + start server)

setup: ## First-time setup (install deps and check)
	@echo "$(GREEN)Setting up FinPilot backend...$(NC)"
	@$(MAKE) install-quick
	@echo ""
	@$(MAKE) check-deps
	@echo ""
	@echo "$(GREEN)Setup complete! Run 'make run' to start the server.$(NC)"

# Debug commands
debug-imports: ## Check if main modules can be imported
	@echo "$(GREEN)Testing module imports...$(NC)"
	@PYTHONPATH=. $(PYTHON) -c "from agents.base_agent import BaseAgent; print('✓ BaseAgent')" 2>&1 || echo "✗ BaseAgent failed"
	@PYTHONPATH=. $(PYTHON) -c "from config import settings; print('✓ Settings')" 2>&1 || echo "✗ Settings failed"
	@PYTHONPATH=. $(PYTHON) -c "from api.contracts import APIContracts; print('✓ API Contracts')" 2>&1 || echo "✗ API Contracts failed"

.DEFAULT_GOAL := help
