.PHONY: help install install-dev test test-cov lint format clean run-examples benchmark

help:
	@echo "NVFP4-DDIM Optimizer - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package and dependencies"
	@echo "  make install-dev    Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linting checks"
	@echo "  make format         Format code with black"
	@echo ""
	@echo "Examples:"
	@echo "  make run-examples   Run all example scripts"
	@echo "  make benchmark      Run performance benchmarks"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and cache"

install:
	@echo "Installing NVFP4-DDIM Optimizer..."
	pip install -e .

install-dev:
	@echo "Installing with development dependencies..."
	pip install -e .
	pip install -r requirements-dev.txt

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src/nvfp4_ddim_optimizer --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Running linting checks..."
	flake8 src/nvfp4_ddim_optimizer tests --max-line-length=100
	mypy src/nvfp4_ddim_optimizer --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src/nvfp4_ddim_optimizer tests examples --line-length=100

run-examples:
	@echo "Running example scripts..."
	@echo "Note: Examples will be available after implementation"
	# python examples/basic_optimization.py
	# python examples/preset_comparison.py
	# python examples/quantization_demo.py

benchmark:
	@echo "Running performance benchmarks..."
	@echo "Note: Benchmarks will be available after implementation"
	# python benchmarks/nvfp4_ddim_benchmark.py

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"
