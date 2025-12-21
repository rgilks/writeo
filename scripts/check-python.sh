#!/bin/bash
# Python code quality checks
# Usage: ./scripts/check-python.sh [format|lint|type-check|all]

set -e

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

check_command() {
    if ! command -v "$1" > /dev/null 2>&1; then
        echo "${RED}Error: $1 not found${NC}"
        echo "${YELLOW}Install with: uv pip install $1${NC}"
        exit 1
    fi
}

run_format() {
    echo "${GREEN}Formatting Python code with ruff...${NC}"
    check_command ruff
    ruff format services/modal-deberta services/modal-lt packages/shared/py
    echo "${GREEN}✓ Formatting complete${NC}"
}

run_format_check() {
    echo "${GREEN}Checking Python code formatting...${NC}"
    check_command ruff
    if ruff format --check services/modal-deberta services/modal-lt packages/shared/py; then
        echo "${GREEN}✓ Formatting check passed${NC}"
    else
        echo "${RED}✗ Formatting check failed. Run 'npm run format:py' to fix.${NC}"
        exit 1
    fi
}

run_lint() {
    echo "${GREEN}Linting Python code with ruff...${NC}"
    check_command ruff
    if ruff check services/modal-deberta services/modal-lt packages/shared/py; then
        echo "${GREEN}✓ Linting passed${NC}"
    else
        echo "${RED}✗ Linting failed${NC}"
        exit 1
    fi
}

run_type_check() {
    echo "${GREEN}Running Python type checker (mypy)...${NC}"
    check_command mypy
    
    ERRORS=0
    
    if [ -d "services/modal-deberta" ]; then
        echo "${YELLOW}Checking services/modal-deberta...${NC}"
        if ! mypy services/modal-deberta --config-file services/modal-deberta/pyproject.toml; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
    
    if [ -d "services/modal-lt" ]; then
        echo "${YELLOW}Checking services/modal-lt...${NC}"
        if ! mypy services/modal-lt --config-file services/modal-lt/pyproject.toml; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
    
    if [ -d "packages/shared/py" ]; then
        echo "${YELLOW}Checking packages/shared/py...${NC}"
        if ! mypy packages/shared/py --config-file packages/shared/py/pyproject.toml; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
    
    if [ $ERRORS -eq 0 ]; then
        echo "${GREEN}✓ Type checking passed${NC}"
    else
        echo "${RED}✗ Type checking failed in $ERRORS directory(ies)${NC}"
        exit 1
    fi
}

case "${1:-all}" in
    format)
        run_format
        ;;
    format-check)
        run_format_check
        ;;
    lint)
        run_lint
        ;;
    type-check)
        run_type_check
        ;;
    all)
        run_format_check
        run_lint
        run_type_check
        echo ""
        echo "${GREEN}All Python checks passed!${NC}"
        ;;
    *)
        echo "Usage: $0 [format|format-check|lint|type-check|all]"
        exit 1
        ;;
esac

