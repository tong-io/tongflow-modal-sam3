"""Modal download entry for SAM 3 / SAM 3.1.

Run:
  modal run download.py::download

Prewarms the gated SAM 3 (image) and SAM 3.1 (video multiplex) checkpoints
into the shared HF cache on the ``models`` volume.

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import os

import modal

REPOS = [
    "facebook/sam3",
    "facebook/sam3.1",
]

volume = modal.Volume.from_name("models", create_if_missing=True)
secrets = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==1.6.0")
    .env({"HF_HOME": "/models/hf"}),
    volumes={"/models": volume},
    secrets=[secrets],
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    for repo_id in REPOS:
        snapshot_download(repo_id=repo_id)
        print(f"Cached {repo_id}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
