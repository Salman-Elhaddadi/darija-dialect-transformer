"""Fetch the MARBERT checkpoint into models/marbert-base/.

`AutoModel.from_pretrained("UBC-NLP/MARBERT")` is the normal way to do this and
works fine on a fast link. It was not usable here: MARBERT ships weights only as
a 654MB pytorch_model.bin (no safetensors shard), and on a ~27KB/s single-stream
connection the standard downloader stalled at 0 bytes and then crawled at ~5KB/s
— a >30h ETA. Eight parallel range requests reach ~90KB/s on the same link
(~2h), which is the connection's real ceiling: 16 streams gave no further gain.

So this does chunked parallel download with per-chunk resume, then verifies the
assembled file against the checkpoint's published sha256 before it is trusted.
Interrupt and re-run it as often as you like; completed chunks are not refetched.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "https://huggingface.co/UBC-NLP/MARBERT/resolve/main"
DEST = Path("models/marbert-base")
SMALL_FILES = ["config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]

WEIGHTS = "pytorch_model.bin"
WEIGHTS_BYTES = 654186400
WEIGHTS_SHA256 = "801862ace5fc9e6b135b19476dbda5c7de1e8673b5b9d55db08d79e2c30b9dca"
N_STREAMS = 8


def curl(url: str, dest: Path, byte_range: str | None = None, append: bool = False) -> None:
    cmd = ["curl", "-sL", "--max-time", "600", "--connect-timeout", "30"]
    if byte_range:
        cmd += ["-r", byte_range]
    cmd.append(url)
    mode = "ab" if append else "wb"
    with open(dest, mode) as fh:
        subprocess.run(cmd, stdout=fh, check=False)


def chunk_bounds(i: int) -> tuple[int, int]:
    size = -(-WEIGHTS_BYTES // N_STREAMS)
    return i * size, min((i + 1) * size, WEIGHTS_BYTES) - 1


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if not (DEST / name).exists():
            curl(f"{BASE_URL}/{name}", DEST / name)
            print(f"got {name} ({(DEST / name).stat().st_size} bytes)")

    target = DEST / WEIGHTS
    if target.exists() and target.stat().st_size == WEIGHTS_BYTES:
        print(f"{target} already complete")
        return

    parts = DEST / "parts"
    parts.mkdir(exist_ok=True)

    def fetch(i: int) -> None:
        start, end = chunk_bounds(i)
        part = parts / f"part_{i}"
        have = part.stat().st_size if part.exists() else 0
        if have >= end - start + 1:
            return
        curl(f"{BASE_URL}/{WEIGHTS}", part, f"{start + have}-{end}", append=True)

    for attempt in range(1, 61):
        with ThreadPoolExecutor(max_workers=N_STREAMS) as pool:
            list(pool.map(fetch, range(N_STREAMS)))

        got = sum((parts / f"part_{i}").stat().st_size for i in range(N_STREAMS) if (parts / f"part_{i}").exists())
        print(f"  round {attempt}: {got * 100 // WEIGHTS_BYTES}% ({got}/{WEIGHTS_BYTES} bytes)")
        if got >= WEIGHTS_BYTES:
            break
    else:
        sys.exit("download did not complete after 60 rounds")

    with open(target, "wb") as out:
        for i in range(N_STREAMS):
            out.write((parts / f"part_{i}").read_bytes())

    actual = sha256(target)
    if actual != WEIGHTS_SHA256:
        target.unlink()
        sys.exit(f"checksum mismatch: {actual} != {WEIGHTS_SHA256}")

    for i in range(N_STREAMS):
        (parts / f"part_{i}").unlink()
    parts.rmdir()
    print(f"verified {target} ({WEIGHTS_BYTES} bytes, sha256 ok)")


if __name__ == "__main__":
    main()
