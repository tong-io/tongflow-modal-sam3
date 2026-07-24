"""Modal deploy entry for SAM 3 / SAM 3.1 (Meta, promptable concept segmentation).

Implements two ABI slots as text-guided matting:
  - ``image-edit``: segment every instance of the concept named by ``text``
    and return the cutout as a transparent PNG (background removed).
  - ``video-edit``: track every instance of the concept named by ``text``
    through the video (SAM 3.1 Object Multiplex) and return a green-screen
    matte with the original audio kept.

The ``facebook/sam3`` / ``facebook/sam3.1`` checkpoints are gated on Hugging
Face: accept the terms on the repo pages, then put that account's
``HF_TOKEN`` in TongFlow Settings before first use.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.video_edit import VideoEditInput, VideoEditOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path, prompt_media_to_bytes
from tongflow.slots import node_slot

REPO_URL = "https://github.com/facebookresearch/sam3.git"
# Pin the upstream revision so redeploys are reproducible (main moves).
REPO_REV = "5dd401d1c5c1d5c3eedff06d41b77af824517619"

# Plugin-internal knobs — NOT ABI fields.
CONFIDENCE = float(os.environ.get("SAM3_CONFIDENCE", 0.5))
# Whole videos are decoded into memory by the predictor; cap length/size so a
# long upload degrades into a clear error instead of an OOM.
MAX_VIDEO_SECONDS = float(os.environ.get("SAM3_MAX_VIDEO_SECONDS", 60))
MAX_VIDEO_WIDTH = int(os.environ.get("SAM3_MAX_VIDEO_WIDTH", 1280))
# Chroma-key color painted where the concept is NOT (B, G, R).
MATTE_BGR = (0, 255, 0)

volume = modal.Volume.from_name("models", create_if_missing=True)
# Both checkpoint repos are gated: forward the local HF_TOKEN (TongFlow
# Settings) into the container so snapshot downloads authenticate.
secrets = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "opencv-python-headless",
        "decord",
        "pillow",
        "einops",
        "pycocotools",
        # sam3.model_builder does `import pkg_resources` at module top;
        # python 3.12 images don't ship setuptools by default, and
        # setuptools>=81 removed pkg_resources entirely.
        "setuptools<81",
    )
    .pip_install(f"git+{REPO_URL}@{REPO_REV}")
    .pip_install("tongflow==0.2.16", "fastapi[standard]")
    .env({"HF_HOME": "/models/hf"})
)

with image.imports():
    import io
    import subprocess
    import tempfile

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import (
        build_sam3_image_model,
        build_sam3_multiplex_video_predictor,
    )


@deploy
@app.cls(
    image=image,
    gpu="L40S",
    memory=32768,
    volumes={"/models": volume},
    secrets=[secrets],
    timeout=3600,
    scaledown_window=5,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Load the SAM 3 image model once; the (heavier) SAM 3.1 video
        predictor is built lazily on the first video call."""
        self.image_model = build_sam3_image_model()
        self.processor = Sam3Processor(
            self.image_model, confidence_threshold=CONFIDENCE
        )
        self._video_predictor = None

    def _video(self):
        if self._video_predictor is None:
            # use_fa3 needs FlashAttention 3 (Hopper); L40S runs without it.
            self._video_predictor = build_sam3_multiplex_video_predictor(
                use_fa3=False
            )
        return self._video_predictor

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        """Cut out every instance of the concept named by ``text``."""
        try:
            phrase = (input.text or "").strip()
            if not phrase:
                raise RuntimeError("text prompt is required")
            img = Image.open(io.BytesIO(prompt_media_to_bytes(input.image)))
            img = img.convert("RGB")

            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(img)
                state = self.processor.set_text_prompt(state=state, prompt=phrase)

            masks = state["masks"]  # bool (N, 1, H, W) at original size
            if masks.shape[0] == 0:
                raise RuntimeError(f"no instance of {phrase!r} found in the image")
            union = masks.any(dim=0)[0].cpu().numpy()  # (H, W) bool

            rgba = np.dstack(
                [np.asarray(img), (union * 255).astype(np.uint8)]
            )
            ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            if not ok:
                raise RuntimeError("PNG encoding failed")
        except Exception as e:
            return ImageEditOutput(success=False, error=str(e))
        return ImageEditOutput(
            success=True, image=asset(png.tobytes(), mime="image/png")
        )

    @modal.method()
    @node_slot(NodeSlots.VIDEO_EDIT)
    def video_edit(self, input: VideoEditInput) -> VideoEditOutput:
        """Track the concept named by ``text``; green-screen everything else."""
        import contextlib

        try:
            phrase = (input.text or "").strip()
            if not phrase:
                raise RuntimeError("text prompt is required")

            with contextlib.ExitStack() as stack:
                src = str(stack.enter_context(asset_as_path(input.video)))
                mp4 = self._normalize_video(stack, src)
                frame_masks = self._track(mp4, phrase)
                out = self._composite(stack, mp4, src, frame_masks)
        except Exception as e:
            return VideoEditOutput(success=False, error=str(e))
        return VideoEditOutput(success=True, video=asset(out, mime="video/mp4"))

    @staticmethod
    def _normalize_video(stack, src: str) -> str:
        """Re-encode the input to a bounded H.264 MP4 the predictor can read."""
        fd, mp4 = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        stack.callback(lambda: os.path.exists(mp4) and os.unlink(mp4))
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", src,
                "-t", str(MAX_VIDEO_SECONDS),
                "-vf", f"scale='min({MAX_VIDEO_WIDTH},iw)':-2",
                "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                mp4,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-300:]
            raise RuntimeError(f"could not decode video input: {tail}")
        return mp4

    def _track(self, mp4: str, phrase: str) -> dict[int, "np.ndarray"]:
        """Run SAM 3.1 multiplex tracking; frame index -> union bool mask."""
        predictor = self._video()
        resp = predictor.handle_request(
            dict(type="start_session", resource_path=mp4)
        )
        sid = resp["session_id"]
        try:
            resp = predictor.handle_request(
                dict(
                    type="add_prompt",
                    session_id=sid,
                    frame_index=0,
                    text=phrase,
                )
            )
            frame_masks: dict[int, np.ndarray] = {}

            def _fold(frame_index: int, outputs) -> None:
                m = outputs["out_binary_masks"]
                if m is not None and len(m) > 0:
                    frame_masks[frame_index] = np.asarray(m).any(axis=0)

            for r in predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=sid)
            ):
                _fold(r["frame_index"], r["outputs"])
            if not frame_masks:
                raise RuntimeError(f"no instance of {phrase!r} found in the video")
            return frame_masks
        finally:
            predictor.handle_request(dict(type="close_session", session_id=sid))

    @staticmethod
    def _composite(stack, mp4: str, original: str, frame_masks) -> bytes:
        """Paint MATTE_BGR outside the mask; mux back the original audio."""
        cap = cv2.VideoCapture(mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        stack.callback(lambda: os.path.exists(out_path) and os.unlink(out_path))

        enc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", f"{fps}",
                "-i", "pipe:0",
                "-i", original,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                out_path,
            ],
            stdin=subprocess.PIPE,
        )
        assert enc.stdin is not None
        backdrop = np.empty((h, w, 3), dtype=np.uint8)
        backdrop[:] = MATTE_BGR
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                mask = frame_masks.get(idx)
                if mask is None:
                    frame = backdrop
                else:
                    m = mask[..., None]
                    frame = np.where(m, frame, backdrop)
                enc.stdin.write(np.ascontiguousarray(frame).tobytes())
                idx += 1
        finally:
            cap.release()
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError("video encoding failed")
        with open(out_path, "rb") as f:
            return f.read()

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

