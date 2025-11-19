# Scripts

Utility scripts for deployment and operations.

## Deployment Scripts

- `deploy-all.sh` - Deploy all services (Modal + API + Web)
- `deploy-modal.sh` - Deploy Modal services only
- `setup.sh` - Initial Cloudflare resource setup (R2 bucket, KV namespace)
- `check-logs.sh` - View Cloudflare Worker logs safely (with timeout)

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
