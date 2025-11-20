# Scripts Reference

Utility scripts for deployment and operations.

## Deployment Scripts

- `deploy-all.sh` - Deploy all services (Modal + API + Web)
- `deploy-modal.sh` - Deploy Modal services only
- `setup.sh` - Initial Cloudflare resource setup (R2 bucket, KV namespace)
- `check-logs.sh` - View Cloudflare Worker logs safely (with timeout)
- `clear-cloudflare-data.sh` - Clear all data from Cloudflare R2 and KV storage (fast bucket recreation)
- `set-mode.sh` - Switch between Cheap Mode and Turbo Mode (local development)
- `set-mode-production.sh` - Switch between Cheap Mode and Turbo Mode (production)

## Mode Switching Scripts

**Local Development:**

```bash
./scripts/set-mode.sh cheap   # Switch to Cheap Mode
./scripts/set-mode.sh turbo   # Switch to Turbo Mode
```

**Production:**

```bash
./scripts/set-mode-production.sh cheap
./scripts/set-mode-production.sh turbo
```

See [MODES.md](MODES.md) for detailed mode switching guide.

## Data Management

**Clear Cloudflare Data:**

```bash
# Make sure you're logged in:
wrangler login

# Then run:
./scripts/clear-cloudflare-data.sh
```

**How it works:**

- ✅ **R2**: Deletes the entire bucket and recreates it (instant, regardless of object count)
- ✅ **KV**: Deletes the entire namespace and recreates it (instant, regardless of key count)
- ✅ **Auth**: Automatically uses Wrangler's stored OAuth token
- ✅ **Auto-update**: Automatically updates `wrangler.toml` with the new KV namespace ID

⚠️ **Warning:** This script permanently deletes all data from R2 buckets and KV namespaces. Use with caution!

## Calibration Scripts (Optional)

- `calibrate-from-corpus.py` - Generate calibration data from corpus
- `apply-corpus-calibration.py` - Generate calibration function code

**Usage:**

```bash
export CORPUS_PATH=/path/to/corpus.tsv
export MODAL_URL=https://your-endpoint.modal.run
python scripts/calibrate-from-corpus.py
```

## References

- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [OPERATIONS.md](OPERATIONS.md) - Operations guide
- [MODES.md](MODES.md) - Mode switching guide
