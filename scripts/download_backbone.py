#!/usr/bin/env python3
"""Download or verify the exact public timm DINOv3 weight artifact."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPOSITORY = "timm/vit_base_patch16_dinov3.lvd1689m"
REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
EXPECTED_SHA256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"


def verify(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("Backbone SHA-256 differs from the recorded main-table artifact")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if not output.exists():
        if args.verify_only:
            raise FileNotFoundError(output)
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(repo_id=REPOSITORY, revision=REVISION,
                                     filename="model.safetensors"))
        verify(cached)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves any file written concurrently.
        with cached.open("rb") as source, output.open("xb") as destination:
            shutil.copyfileobj(source, destination)
    digest = verify(output)
    print(json.dumps({"path": str(output), "repository": REPOSITORY,
                      "revision": REVISION, "sha256": digest, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
