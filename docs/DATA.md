# Data

What we actually hold, as found. Exact file layouts are added in Phase 1 (plan §7).

## ISLES'24

Stored at `StrokeCopilot/raw/isles24_train.7z` on the shared Drive. The md5 was computed on the copy stored on Drive, and it matches the value on Zenodo.

```json
{
  "record_id": 17652035,
  "doi": "10.5281/zenodo.17652035",
  "file": "train.7z",
  "size_bytes": 99022114670,
  "md5": "4959a5dd2438d53e3c86d6858484e781",
  "verified_utc": "2026-10-31T16:28:51.820341+00:00"
}
```

The archive has not been opened yet. Licence: CC BY-NC-SA 4.0 (plan §5.1).

## RSNA ICH

Built on Kaggle from the competition's stage 2 training data into private datasets (DEC-17): `rsna_slim_v1`, 320 px, JPEG quality 95, 4 shards. 752,803 labelled slices in total.

| Shard | Dataset | Slices | Failed | Tar size (GiB) | Status |
|---|---|---:|---:|---:|---|
| 0 | `rsna-slim-v1-s0` | 188,251 | 0 | 4.371 | Built and verified on Kaggle |
| 1 | `rsna-slim-v1-s1` | | | | Not recorded yet |
| 2 | `rsna-slim-v1-s2` | | | | Not recorded yet |
| 3 | `rsna-slim-v1-s3` | | | | Not recorded yet |

Shard 0 projects 16.94 GiB for all four shards. Reading speed from an attached dataset has not been measured. Excluded scans: none so far.
