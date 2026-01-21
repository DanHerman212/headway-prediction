#!/bin/bash
# Pre-deployment validation script
# Run this before EVERY deployment to catch common issues

set -e  # Exit on any error

echo "========================================================================"
echo "PRE-DEPLOYMENT VALIDATION"
echo "========================================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ERRORS=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ERRORS=$((ERRORS + 1))
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 1. Check Python environment
echo "1. Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found"
else
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_success "Python found: $PYTHON_VERSION"
fi
echo ""

# 2. Check required modules can be imported
echo "2. Checking Python imports..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

errors = []
try:
    from src import config
    print('  ✓ config')
except Exception as e:
    errors.append(f'config: {e}')

try:
    from src import preprocess
    print('  ✓ preprocess')
except Exception as e:
    errors.append(f'preprocess: {e}')

try:
    from src import train
    print('  ✓ train')
except Exception as e:
    errors.append(f'train: {e}')

try:
    from src import model
    print('  ✓ model')
except Exception as e:
    errors.append(f'model: {e}')

if errors:
    print('\\nImport errors:')
    for error in errors:
        print(f'  ✗ {error}')
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))
echo ""

# 3. Check configuration
echo "3. Validating configuration..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.config import config

errors = []

# Required configs
required = {
    'BQ_PROJECT': config.BQ_PROJECT,
    'BQ_DATASET': config.BQ_DATASET,
    'BQ_TABLE': config.BQ_TABLE,
    'BATCH_SIZE': config.BATCH_SIZE,
    'EPOCHS': config.EPOCHS,
    'EXPERIMENT_NAME': config.EXPERIMENT_NAME,
}

for name, value in required.items():
    if value is None or value == '':
        errors.append(f'{name} is not set')
    else:
        print(f'  ✓ {name}: {value}')

# Validate numeric configs
if config.BATCH_SIZE <= 0:
    errors.append('BATCH_SIZE must be positive')
if config.EPOCHS <= 0:
    errors.append('EPOCHS must be positive')
if not (0 < config.TRAIN_SPLIT < 1):
    errors.append('TRAIN_SPLIT must be between 0 and 1')

if errors:
    print('\\nConfiguration errors:')
    for error in errors:
        print(f'  ✗ {error}')
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))
echo ""

# 4. Run unit tests
echo "4. Running unit tests..."
if [ -d "tests" ]; then
    if command -v python3 &> /dev/null; then
        python3 -m pytest tests/test_dataset.py tests/test_preprocess.py tests/test_model.py tests/test_vertex_callback.py -v --tb=short 2>/dev/null || ERRORS=$((ERRORS + 1))
    else
        print_warning "python3 not found, skipping unit tests"
    fi
else
    print_warning "tests/ directory not found, skipping unit tests"
fi
echo ""

# 5. Run integration test
echo "5. Running integration test..."
if [ -f "tests/test_integration.py" ]; then
    if command -v python3 &> /dev/null; then
        python3 -m pytest tests/test_integration.py::test_full_pipeline_simulation -v --tb=short 2>/dev/null || ERRORS=$((ERRORS + 1))
    else
        print_warning "python3 not found, skipping integration test"
    fi
else
    print_warning "Integration test not found"
fi
echo ""

# 6. Check for common mistakes
echo "6. Checking for common mistakes..."

# Check that train.py uses drop_remainder
if grep -q "\.batch(config\.BATCH_SIZE, drop_remainder=True)" src/train.py; then
    print_success "drop_remainder=True found in batch calls"
else
    print_warning "drop_remainder=True not found - may cause batch size mismatch"
fi

# Check that Vertex AI Experiments is initialized with experiment parameter
if grep -q "experiment=" src/train.py; then
    print_success "Vertex AI experiment parameter found"
else
    print_warning "Vertex AI experiment parameter not found"
fi

# Check that artifacts are handled as directories
if grep -q "Path(.*\.path) /" pipeline.py; then
    print_success "Artifact directory handling looks correct"
else
    print_warning "Artifact handling may not treat paths as directories"
fi

echo ""

# 7. Check Docker configuration
echo "7. Checking Docker configuration..."
if [ -f "Dockerfile" ]; then
    print_success "Dockerfile found"
    
    # Check for common issues
    if grep -q "COPY src/" Dockerfile; then
        print_success "Dockerfile copies src/"
    else
        print_warning "Dockerfile may not copy src/ directory"
    fi
else
    print_error "Dockerfile not found"
fi
echo ""

# 8. Check deployment script
echo "8. Checking deployment script..."
if [ -f "deploy.sh" ]; then
    print_success "deploy.sh found"
    if [ -x "deploy.sh" ]; then
        print_success "deploy.sh is executable"
    else
        print_warning "deploy.sh is not executable (chmod +x deploy.sh)"
    fi
else
    print_error "deploy.sh not found"
fi
echo ""

# Summary
echo "========================================================================"
if [ $ERRORS -eq 0 ]; then
    print_success "ALL CHECKS PASSED - READY FOR DEPLOYMENT"
    echo ""
    echo "To deploy, run:"
    echo "  ./deploy.sh <run_name>"
    echo ""
    exit 0
else
    print_error "VALIDATION FAILED WITH $ERRORS ERROR(S)"
    echo ""
    echo "Fix the errors above before deploying."
    echo ""
    exit 1
fi
