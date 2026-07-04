# tongflow-modal-sam3

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text-guided image/video matting with **SAM 3 / SAM 3.1** (Meta, `facebook/sam3` + `facebook/sam3.1` Object Multiplex), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Image editing** (`image-edit`) — describe a concept ("the dog", "players in white") and get every instance cut out as a transparent PNG.
- **Video editing** (`video-edit`) — track every instance of the concept through the video (SAM 3.1 Object Multiplex, ~7x faster multi-object tracking) and return a green-screen matte, original audio kept.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ✅ | The checkpoints are gated: request access on [facebook/sam3](https://huggingface.co/facebook/sam3) and [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), then use that account's token. |

On first use the plugin deploys to your Modal account automatically and caches the build; weights are cached on a shared Modal volume.

## Tuning (env, optional)

| Env | Default | Notes |
| --- | --- | --- |
| `SAM3_CONFIDENCE` | `0.5` | Detection confidence threshold. |
| `SAM3_MAX_VIDEO_SECONDS` | `60` | Videos are trimmed to this length. |
| `SAM3_MAX_VIDEO_WIDTH` | `1280` | Videos are downscaled to at most this width. |
