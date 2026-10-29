# data/

Nothing in this folder is committed except this file. All data lives in the shared Google Drive folder `StrokeCopilot/` (plan §4.2). Add a shortcut to it in your own Drive so it mounts under `MyDrive/StrokeCopilot/`.

```
StrokeCopilot/
  raw/        ISLES'24 archive + md5, RSNA archive — untouched after download
  slim/       isles24_slim_v1.tar, rsna_slim_v1.tar — immutable; new version = new file
  runs/       EXP-<id>_<short-name>/  config.json, checkpoints/, metrics.json, logs/
  artifacts/  guidelines/ (PDFs), figures/ (exported for the report), corpus/ (passages.jsonl)
```

Never train by reading individual files through the Drive mount: copy the slim tarball to Colab local disk and extract it there (plan §4.3). Exact archive layouts go in `docs/DATA.md` (Phase 1).
