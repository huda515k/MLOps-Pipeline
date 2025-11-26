.PHONY: help install setup test lint format clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install     - Install Python dependencies"
	@echo "  make setup       - Initial setup (DVC, directories)"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code with Black"
	@echo "  make clean       - Clean temporary files"
	@echo "  make docker-up   - Start all services with Docker Compose"
	@echo "  make docker-down - Stop all services"
	@echo "  make docker-build - Build Docker images"

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed models logs
	dvc init
	@echo "Setup complete. Configure DVC remote with: dvc remote add -d storage <your-remote>"

test:
	pytest tests/ -v

lint:
	flake8 scripts/ src/ dags/ --max-line-length=120 --ignore=E203,W503
	black --check scripts/ src/ dags/

format:
	black scripts/ src/ dags/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

