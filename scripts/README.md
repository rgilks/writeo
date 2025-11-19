# Scripts

Utility scripts for deployment and operations.

## Deployment Scripts

- `deploy-all.sh` - Deploy all services (Modal + API + Web)
- `deploy-modal.sh` - Deploy Modal services only
- `setup.sh` - Initial Cloudflare resource setup (R2 bucket, KV namespace)
- `check-logs.sh` - View Cloudflare Worker logs safely (with timeout)
- `clear-cloudflare-data.sh` - Clear all data from Cloudflare R2 and KV storage (fast bucket recreation)

**Usage for clear-cloudflare-data.sh:**

The script deletes and recreates the R2 bucket for instant clearing:

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

**Note:**

- The R2 bucket binding in `wrangler.toml` will continue to work after recreation since the bucket name stays the same.
- The KV namespace ID in `wrangler.toml` will be automatically updated with the new namespace ID.
- If your worker is currently deployed, you should redeploy it after running this script to use the new namespace ID.

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
