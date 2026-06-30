"""
RAM-safe, resumable OpenWebText downloader.

Bypasses the HuggingFace `datasets` library entirely to avoid its internal
Apache Arrow buffers, which can silently consume all available RAM even with
streaming=True.

Instead we:
  1. Fetch the list of parquet shard files from the HF Hub datasets-server API.
  2. Stream each shard file to disk with requests (no full file in RAM).
  3. Read the shard with pyarrow one batch at a time, write text, then DELETE
     the shard — so disk use for temp files stays low too.
  4. Save a checkpoint after every completed shard so the download can be
     PAUSED (Ctrl+C) and RESUMED safely without re-downloading done shards.

Requirements:
    pip install requests pyarrow tqdm
"""

import os
import gc
import json
import requests
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_REPO    = "Skylion007/openwebtext"
OUTPUT_FILE     = "openwebtext_10gb.txt"
TEMP_DIR        = "parquet_shards"          # temporary folder for shard files
CHECKPOINT_FILE = "download_checkpoint.json"
TARGET_BYTES    = 10 * 1024 ** 3            # 10 GB
PYARROW_BATCH   = 1_000                     # rows per pyarrow read_batch call
# ---------------------------------------------------------------------------

os.makedirs(TEMP_DIR, exist_ok=True)


# ── Checkpoint helpers ───────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: shard {ckpt['last_shard_completed'] + 1}, "
              f"{ckpt['bytes_written'] / 1024**3:.2f} GB already written.\n")
        return ckpt
    return {"last_shard_completed": -1, "bytes_written": 0}


def save_checkpoint(last_shard_idx: int, bytes_written: int) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_shard_completed": last_shard_idx,
                   "bytes_written": bytes_written}, f)


# ── Step 1: get the list of parquet shards from HuggingFace Hub ─────────────
print("Fetching shard list from HuggingFace Hub API...")
api_url = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET_REPO}"
resp = requests.get(api_url, timeout=30)
resp.raise_for_status()
shards = resp.json()["parquet_files"]   # list of {"url": ..., "filename": ...} dicts
print(f"  Found {len(shards)} shards.\n")


# ── Step 2: load checkpoint ──────────────────────────────────────────────────
ckpt = load_checkpoint()
start_shard_idx = ckpt["last_shard_completed"] + 1
bytes_written   = ckpt["bytes_written"]
file_mode       = "a" if bytes_written > 0 else "w"   # append if resuming


def stream_download(url: str, dest: str) -> None:
    """Download a URL to dest using HTTP chunked streaming (low RAM)."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
                f.write(chunk)


# ── Step 3: iterate shards until we hit TARGET_BYTES ────────────────────────
done = False

with tqdm(total=TARGET_BYTES, initial=bytes_written,
          unit="B", unit_scale=True, unit_divisor=1024,
          desc="Total written") as pbar:

    with open(OUTPUT_FILE, file_mode, encoding="utf-8") as out_f:

        for shard_idx, shard_info in enumerate(shards):
            if done:
                break

            # Skip already-completed shards
            if shard_idx < start_shard_idx:
                continue

            shard_url  = shard_info["url"]
            shard_name = os.path.basename(shard_url.split("?")[0])
            shard_path = os.path.join(TEMP_DIR, shard_name)

            # ── download the shard ──────────────────────────────────────────
            print(f"\nDownloading shard: {shard_name}")
            stream_download(shard_url, shard_path)

            # ── read the shard row-batch by row-batch ───────────────────────
            parquet_file = pq.ParquetFile(shard_path)

            for batch in parquet_file.iter_batches(
                batch_size=PYARROW_BATCH, columns=["text"]
            ):
                texts = batch.column("text").to_pylist()

                for text in texts:
                    line = text + "\n\n"
                    out_f.write(line)
                    chunk_size = len(line.encode("utf-8"))
                    bytes_written += chunk_size
                    pbar.update(chunk_size)

                    if bytes_written >= TARGET_BYTES:
                        done = True
                        break

                # Release the batch from RAM immediately
                del texts, batch
                gc.collect()

                if done:
                    break

            # ── done with this shard — delete it to free disk space ─────────
            parquet_file = None          # close file handle
            gc.collect()
            os.remove(shard_path)
            print(f"  Shard processed and deleted.")

            # ── save checkpoint ONLY after a full shard completes ───────────
            if not done:
                save_checkpoint(shard_idx, bytes_written)
                print(f"  Checkpoint saved (shard {shard_idx} done, "
                      f"{bytes_written / 1024**3:.2f} GB total).")

# Clean up temp dir if empty
try:
    os.rmdir(TEMP_DIR)
except OSError:
    pass  # not empty — that's fine

# Remove checkpoint once fully done
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"\nFinished! Wrote {bytes_written / 1024**3:.2f} GB to {OUTPUT_FILE}.")