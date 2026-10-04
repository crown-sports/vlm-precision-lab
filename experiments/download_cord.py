"""Download CORD-v2 parquet sources at the recorded revision; verify every byte."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from urllib.request import getproxies

from precisionlab.data import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["validation"])
    p.add_argument("--mirror", default="https://huggingface.co", help="HTTPS transport; pinned hashes still required")
    a = p.parse_args()
    if not a.mirror.startswith("https://"):
        p.error("Use an HTTPS transport")
    source = json.loads(Path(__file__).with_name("cord-source.json").read_text())
    files = [x for x in source["siblings"] if any(x["rfilename"].startswith("data/" + s + "-") for s in a.splits)]

    def download(item):
        path = a.output / item["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = item["lfs"]["sha256"]
        if path.exists():
            if path.stat().st_size == item["size"] and sha256(path) == expected:
                return {"file": item["rfilename"], "sha256": expected, "cached": True}
            raise ValueError("Existing source differs from the pinned release")
        partial = path.with_suffix(".parquet.part")
        url = a.mirror.rstrip("/") + "/datasets/" + source["id"] + "/resolve/" + source["sha"] + "/" + item["rfilename"]
        transport_env = os.environ.copy()
        # urllib reads macOS system proxy settings; curl does not. Keep auth
        # material in the subprocess environment rather than its command line.
        for scheme in ("http", "https"):
            proxy = getproxies().get(scheme)
            if proxy:
                transport_env.setdefault(scheme + "_proxy", proxy)
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
            "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "15", "--max-time", "1800",
            "--continue-at", "-", "--output", str(partial), url], check=True, env=transport_env)
        if partial.stat().st_size != item["size"] or sha256(partial) != expected:
            raise ValueError("Downloaded CORD source failed pinned SHA256 verification")
        partial.replace(path)
        return {"file": item["rfilename"], "sha256": expected, "cached": False}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(download, files):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
