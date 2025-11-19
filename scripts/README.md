# Scripts

Utility scripts for deployment and operations.

## Deployment Scripts

- `deploy-all.sh` - Deploy all services (Modal + API + Web)
- `deploy-modal.sh` - Deploy Modal services only
- `setup.sh` - Initial Cloudflare resource setup (R2 bucket, KV namespace)
- `check-logs.sh` - View Cloudflare Worker logs safely (with timeout)
- `clear-cloudflare-data.sh` - Clear all data from Cloudflare R2 and KV storage (uses Cloudflare API bulk delete)

**Usage for clear-cloudflare-data.sh:**

The script uses Cloudflare API bulk delete for fast R2 deletion:

```bash
# Make sure you're logged in:
wrangler login

# Then run:
./scripts/clear-cloudflare-data.sh
```

**How it works:**

- ✅ **R2**: Uses Cloudflare API bulk delete endpoint (up to 1000 objects per request) for fast deletion
- ✅ **KV**: Lists and deletes keys individually (KV doesn't support bulk delete)
- ✅ **Auth**: Automatically uses Wrangler's stored OAuth token

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

- [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) - Deployment guide
- [docs/OPERATIONS.md](../docs/OPERATIONS.md) - Operations guide
