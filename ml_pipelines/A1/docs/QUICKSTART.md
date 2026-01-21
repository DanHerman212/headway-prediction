# 🚀 QUICK START GUIDE

## Before Every Deployment - Run This First!

```bash
./pre_deploy.sh
```

**This single script catches 90% of deployment issues in 30 seconds.**

If it passes → Deploy with confidence  
If it fails → Fix the errors shown

## Complete Development Cycle

```bash
# 1. Edit code in src/

# 2. Validate (REQUIRED before deploy)
./pre_deploy.sh

# 3. Deploy (only if validation passed)
./deploy.sh baseline3
```

## What pre_deploy.sh Checks

1. ✓ Python imports work
2. ✓ Configuration is valid
3. ✓ Unit tests pass
4. ✓ Integration test passes (simulates full pipeline)
5. ✓ No common mistakes (batch size, Vertex AI params, artifacts)
6. ✓ Docker setup is correct

**Time**: ~30 seconds  
**Benefit**: Catches issues that would take 20+ minutes to discover on Vertex AI

## Key Commands

```bash
# The most important command - run before deploy
./pre_deploy.sh

# Deploy to Vertex AI
./deploy.sh <run_name>

# Debug artifacts
python debug_tools/inspect_artifact.py <path>

# Test specific component
python debug_tools/test_component.py --component dataset
```

## Golden Rule

**NEVER run `./deploy.sh` without running `./pre_deploy.sh` first.**

Those 30 seconds save hours of debugging.

## New to This Project?

1. Read [WORKFLOW.md](WORKFLOW.md) for complete workflow documentation
2. Run `./pre_deploy.sh` to understand what checks are performed  
3. Look at tests/ to see what's being validated
4. Check debug_tools/ to see available debugging utilities

## Something Failed?

1. Run `./pre_deploy.sh` - it will tell you exactly what's wrong
2. Check [WORKFLOW.md](WORKFLOW.md) - Common Issues section
3. Use debug tools in debug_tools/

## Previous Debugging Session Summary

**Issues it now catches**:
- Batch size mismatches
- Missing Vertex AI configuration
- Artifact handling errors
- Import errors
- Configuration mistakes
- Dataset shape mismatches

## Files You Care About

- **pre_deploy.sh** ← Run this before every deploy
- **WORKFLOW.md** ← Read when you need details
- **src/** ← Edit your ML code here
- **tests/** ← Tests that prevent bugs
- **debug_tools/** ← Tools when things go wrong

---

**Bottom Line**: Run `./pre_deploy.sh` before `./deploy.sh`. Every time. No exceptions.
