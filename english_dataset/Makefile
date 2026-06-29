PYTHON = python3
PIP = pip
MAIN_SCRIPT = main.py

.PHONY: install run debug clean lint lint-strict

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) $(MAIN_SCRIPT)

debug:
	$(PYTHON) -m pdb $(MAIN_SCRIPT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
