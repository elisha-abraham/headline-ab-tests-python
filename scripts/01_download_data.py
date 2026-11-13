"""Download two files of the Upworthy Research Archive into data/raw/.

    python scripts/01_download_data.py

The archive was released in three parts so that analysts can explore one part
and test what they find on another. This project uses the exploratory file to
form hypotheses and the confirmatory file to test them.

Nothing is changed: the files are saved exactly as published, and their sizes
and SHA-256 digests are written to data/raw/manifest.json.

Source: Matias, J. N., Munger, K., Aubin Le Quere, M., & Ebersole, C. (2021).
        The Upworthy Research Archive. https://osf.io/jd64p/
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FILES = {
    "upworthy-archive-exploratory-packages-03.12.2020.csv": "https://osf.io/download/3vqmp/",
    "upworthy-archive-confirmatory-packages-03.12.2020.csv": "https://osf.io/download/vy8mj/",
}


def fetch(url: str, target: Path) -> None:
    """Stream one file to disk, retrying on network errors."""
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; portfolio-analysis/1.0)"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            partial.replace(target)
            return
        except Exception as error:  # noqa: BLE001 - any network failure is retried
            print(f"  retry {attempt + 1}: {type(error).__name__}: {error}", flush=True)
            time.sleep(5 + 5 * attempt)
    raise SystemExit(f"Could not download {url}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    for name, url in FILES.items():
        target = RAW / name
        if not target.exists():
            print(f"downloading {name} ...", flush=True)
            fetch(url, target)
        manifest["files"][name] = {"url": url, "bytes": target.stat().st_size, "sha256": sha256(target)}
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB", flush=True)
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
