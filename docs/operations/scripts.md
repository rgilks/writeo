# Scripts Reference

Utility scripts for deployment, operations, and evaluation.

## Git Hooks

**Install git hooks:**

```bash
npm run install-hooks
# or
./scripts/install-hooks.sh
```

This installs:

- `pre-commit` - Formats code, runs linting and type checking
- `pre-push` - Runs all tests against local servers

See [Testing Guide](testing.md) for details on hook features including quick mode.

## Deployment Scripts

- `setup.sh` - Initial Cloudflare resource setup (R2 bucket, KV namespace)
- `deploy-all.sh` - **Legacy**: Deployment script for older configurations.
- `deploy-modal.sh` - **Legacy**: Deployment script for `modal-essay`.

> [!TIP]
> For modern production deployment, prefer manual individual service deployment as described in [DEPLOYMENT.md](deployment.md).

## Operational Modes

Scripts to switch between **Economy** (Development) and **Production** (Turbo) modes.

**Local Development:**

```bash
./scripts/set-mode.sh cheap   # Switch to Economy Mode
./scripts/set-mode.sh turbo   # Switch to Production Mode
```

**Production:**

```bash
./scripts/set-mode-production.sh cheap
./scripts/set-mode-production.sh turbo
```

See [MODES.md](modes.md) for detailed mode switching guide.

## Operations Scripts

Helper scripts for managing the running application.

### API Key Management

Manage API keys stored in Cloudflare KV.

```bash
# Create a new key
./scripts/manage-api-keys.sh create "Client Name"

# Revoke a key
./scripts/manage-api-keys.sh revoke <key>

# Check key owner
./scripts/manage-api-keys.sh get <key>
```

### Rate Limits

Reset rate limit counters in Cloudflare KV without affecting other data.

```bash
./scripts/reset-rate-limits.sh
```

### Data Management

**Clear Cloudflare Data:**

Permanently delete all data from R2 buckets and KV namespaces (useful for resetting dev/staging environments).

```bash
./scripts/clear-cloudflare-data.sh
```

- ✅ **R2**: Deletes/Recreates bucket
- ✅ **KV**: Deletes/Recreates namespace
- ✅ **Auto-update**: Updates `wrangler.toml`

### Log Checking

Safe wrapper around `wrangler tail` with timeout protection.

```bash
./scripts/check-logs.sh api-worker "error" 20
```

## Evaluation & Testing

Scripts for evaluating model performance and testing.

### Assessor Evaluation

Runs a comparative evaluation of all enabled assessors against the test dataset (`scripts/training/data/test.jsonl`).

```bash
# Evaluate against local API
export API_KEY=your-key
python scripts/evaluate_assessors.py
```

Generates an `assessor_report.md` with:

- Individual essay scores vs Human labels
- MAE (Mean Absolute Error) and Bias metrics
- Performance recommendations

### Latency Measurement

Measure response latency of a specific endpoint (useful for checking Modal cold/warm starts).

```bash
python scripts/measure_latency.py
```

### Test Environment

Starts the API Worker and Web App on test ports (8787/3000) for integration testing.

```bash
./scripts/start-test-env.sh
```

## Calibration Scripts (Optional)

Legacy scripts for calibrating corpus scores.

- `calibrate-from-corpus.py` - Generate calibration data from corpus
- `apply-corpus-calibration.py` - Generate calibration function code

**Usage:**

```bash
export CORPUS_PATH=/path/to/corpus.tsv
export MODAL_URL=https://your-endpoint.modal.run
python scripts/calibrate-from-corpus.py
```

## References

- [DEPLOYMENT.md](deployment.md) - Deployment guide
- [Operations Guide](monitoring.md) - Operations guide
- [MODES.md](modes.md) - Mode switching guide
