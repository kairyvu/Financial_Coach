# Financial Coach - Makefile
# ===========================

.PHONY: help setup install clean run demo evaluate test train cli lint

# Default Python
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Financial Coach - Available Commands$(NC)"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

# ===========================
# Environment Setup
# ===========================

setup: $(VENV)/bin/activate ## Create virtual environment and install dependencies
	@echo "$(GREEN)✓ Environment ready!$(NC)"

$(VENV)/bin/activate:
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(BLUE)Upgrading pip...$(NC)"
	$(PIP) install --upgrade pip
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

install: setup ## Alias for setup

reinstall: clean setup ## Clean and reinstall everything

# ===========================
# Running the Application
# ===========================

run: setup ## Run the demo (default persona: balanced_ben)
	@echo "$(BLUE)Running Financial Coach Demo...$(NC)"
	$(PYTHON_VENV) demo.py

demo: setup ## Run interactive demo (use PERSONA=name to specify)
	@echo "$(BLUE)Running demo for $(or $(PERSONA),balanced_ben)...$(NC)"
	$(PYTHON_VENV) demo.py --persona $(or $(PERSONA),balanced_ben)

demo-all: setup ## Run demo for all personas
	@echo "$(BLUE)Running demo for all personas...$(NC)"
	$(PYTHON_VENV) demo.py --all

# ===========================
# Training & Evaluation
# ===========================

train: setup ## Train model for a persona (use PERSONA=name)
	@echo "$(BLUE)Training model for $(or $(PERSONA),balanced_ben)...$(NC)"
	$(PYTHON_VENV) cli.py train $(or $(PERSONA),balanced_ben)

train-all: setup ## Train models for all personas
	@echo "$(BLUE)Training models for all personas...$(NC)"
	@for persona in frugal_frank spender_sarah balanced_ben student_sam executive_emma; do \
		echo "$(YELLOW)Training $$persona...$(NC)"; \
		$(PYTHON_VENV) cli.py train $$persona; \
	done
	@echo "$(GREEN)✓ All models trained!$(NC)"

evaluate: setup ## Run full evaluation across all personas
	@echo "$(BLUE)Running evaluation...$(NC)"
	$(PYTHON_VENV) -m src.evaluation.run_evaluation
	@echo "$(GREEN)✓ Results saved to results/$(NC)"

# ===========================
# CLI Commands
# ===========================

cli: setup ## Run CLI (use CMD="command" for specific commands)
	$(PYTHON_VENV) cli.py $(CMD)

personas: setup ## List all available personas
	$(PYTHON_VENV) cli.py personas

status: setup ## Check status of trained models
	$(PYTHON_VENV) cli.py status

score: setup ## Score a transaction (use PERSONA, AMOUNT, CATEGORY, HOUR, DAY, DAYS_TO_END)
	$(PYTHON_VENV) cli.py score $(or $(PERSONA),balanced_ben) \
		--amount $(or $(AMOUNT),100) \
		--category $(or $(CATEGORY),dining) \
		--hour $(or $(HOUR),19) \
		--day $(or $(DAY),3) \
		--days-to-end $(or $(DAYS_TO_END),15)

replay: setup ## Replay transactions for a persona (use PERSONA=name, DAYS=n)
	$(PYTHON_VENV) cli.py replay $(or $(PERSONA),balanced_ben) --days $(or $(DAYS),30)

# ===========================
# MCP Server
# ===========================

server: setup ## Start the MCP Accountant server
	@echo "$(BLUE)Starting MCP Accountant Server...$(NC)"
	$(PYTHON_VENV) -m src.mcp_server.accountant

# ===========================
# Testing & Quality
# ===========================

test: setup ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON_VENV) -m pytest tests/ -v

test-quick: setup ## Run tests without verbose output
	$(PYTHON_VENV) -m pytest tests/

lint: setup ## Check code with basic linting
	@echo "$(BLUE)Checking code...$(NC)"
	$(PYTHON_VENV) -m py_compile src/**/*.py demo.py cli.py
	@echo "$(GREEN)✓ No syntax errors$(NC)"

# ===========================
# Data Generation
# ===========================

generate-data: setup ## Generate synthetic data for all personas
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	$(PYTHON_VENV) -c "from src.data.generator import generate_all_personas_dataset, save_datasets; save_datasets(generate_all_personas_dataset(days=75))"
	@echo "$(GREEN)✓ Data saved to data/synthetic/$(NC)"

# ===========================
# Cleanup
# ===========================

clean: ## Remove virtual environment and caches
	@echo "$(YELLOW)Cleaning up...$(NC)"
	rm -rf $(VENV)
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	@echo "$(GREEN)✓ Cleaned!$(NC)"

clean-artifacts: ## Remove trained model artifacts
	@echo "$(YELLOW)Removing artifacts...$(NC)"
	rm -rf artifacts/
	@echo "$(GREEN)✓ Artifacts removed!$(NC)"

clean-results: ## Remove evaluation results
	@echo "$(YELLOW)Removing results...$(NC)"
	rm -rf results/
	@echo "$(GREEN)✓ Results removed!$(NC)"

clean-all: clean clean-artifacts clean-results ## Remove everything (env, artifacts, results)
	@echo "$(GREEN)✓ All cleaned!$(NC)"

# ===========================
# Quick Start
# ===========================

quickstart: setup train-all demo-all evaluate ## Full setup: install, train all, demo all, evaluate
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  Financial Coach - Setup Complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Try these commands:"
	@echo "  make demo PERSONA=spender_sarah"
	@echo "  make score PERSONA=balanced_ben AMOUNT=200 CATEGORY=shopping"
	@echo "  make replay PERSONA=student_sam DAYS=60"
	@echo ""

