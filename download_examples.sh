#!/bin/bash
# download_examples.sh
# Usage: ./download_examples.sh output_dir

python3 - <<'PYCODE'
from huggingface_hub import snapshot_download
import os

repo_id = "Azorgz/XCalib"
folder = "examples"
local_dir = os.getcwd()

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{folder}/*"],
    local_dir=local_dir
)

print("✔ Downloaded folder:", local_dir)
PYCODE