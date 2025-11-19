# Scripts

Utility scripts for deployment and operations.

## Deployment Scripts

- `deploy-all.sh` - Deploy all services (Modal + API + Web)
- `deploy-modal.sh` - Deploy Modal services only
- `setup.sh` - Initial Cloudflare resource setup (R2 bucket, KV namespace)
- `check-logs.sh` - View Cloudflare Worker logs safely (with timeout)
- `clear-cloudflare-data.sh` - Clear all data from Cloudflare R2 and KV storage

**Usage for clear-cloudflare-data.sh:**

The script automatically uses your Wrangler authentication if you're logged in:

```bash
# If you're already logged in with wrangler login, just run:
./scripts/clear-cloudflare-data.sh

# Or manually set an API token (optional):
export CLOUDFLARE_API_TOKEN='your-token-here'
./scripts/clear-cloudflare-data.sh
```

**Authentication:**

- ✅ **Automatic**: Uses Wrangler's stored OAuth token (from `wrangler login`)
- ✅ **Manual**: Set `CLOUDFLARE_API_TOKEN` environment variable if needed

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
