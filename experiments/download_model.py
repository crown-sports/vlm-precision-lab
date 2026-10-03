"""Download a pinned HF manifest, verifying LFS SHA256 or Git blob IDs.

ModelScope is an optional transport mirror; it is accepted only when its bytes
match the pinned Hugging Face revision. No weights are bundled in this repo.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def matches(path, info):
    if not path.exists() or path.stat().st_size != info["size"]:
        return False
    lfs = info.get("lfs", {})
    if lfs.get("sha256"):
        return digest(path) == lfs["sha256"]
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == info["blobId"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope")
    a = p.parse_args()
    meta = json.loads(Path(__file__).with_name("model-source.json").read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    def fetch(info):
        name = info["rfilename"]
        if Path(name).name != name or name.startswith("."):
            raise ValueError("Only flat, visible files supported")
        target = a.output / name
        if not matches(target, info):
            if target.exists():
                raise ValueError(f"Existing file differs from pinned model: {name}")
            partial = target.with_name(name + ".part")
            if a.source == "modelscope":
                query = urlencode({"Revision": "master", "FilePath": name})
                url = f"https://modelscope.cn/api/v1/models/{meta['id']}/repo?{query}"
            else:
                url = f"https://huggingface.co/{meta['id']}/resolve/{meta['sha']}/{name}"
            print(json.dumps({"file": name, "state": "downloading", "bytes": info["size"]}), flush=True)
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                            "--connect-timeout", "15", "--max-time", "1800", "--continue-at", "-",
                            "--output", str(partial), url], check=True)
            if not matches(partial, info):
                raise ValueError(f"Pinned model hash mismatch: {name}")
            partial.replace(target)
        print(json.dumps({"file": name, "state": "verified"}), flush=True)
        return {"file": name, "bytes": target.stat().st_size, "sha256": digest(target)}

    infos = [i for i in meta["siblings"] if i["rfilename"] != ".gitattributes"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, infos))
    (a.output / "download-manifest.json").write_text(json.dumps({"repository": meta["id"], "revision": meta["sha"],
        "transport": a.source, "files": files, "seconds": time.monotonic() - start}, indent=2) + "\n")


if __name__ == "__main__":
    main()
