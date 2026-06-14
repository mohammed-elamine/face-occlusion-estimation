# MediaPipe models

Home for the **MediaPipe Face Landmarker** model used by the synthetic-occlusion and
face-mask pipelines (`MediaPipeFaceRegionProvider`). The model weights are **git-ignored**
(`*.task`); only this README is tracked.

## `face_landmarker.task`

- **Used when** the installed MediaPipe lacks `mp.solutions` (the Tasks `FaceLandmarker`
  backend). When `mp.solutions` is present, the legacy `solutions.face_mesh` backend bundles
  its model inside the pip package and this file is not needed.
- **Source:** `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- **Get it:**
  ```bash
  make mediapipe-model        # downloads here
  # or:  uv run python -m scripts.data.download_mediapipe_model
  ```
  The builders (`build_face_masks`, `build_synthetic_cache`) also auto-download it here on
  demand. Override the location with `FACE_OCCLUSION_MEDIAPIPE_FACE_LANDMARKER=<path>`.

Search order (`_resolve_mediapipe_model_asset_path`): the env var, then
`models/mediapipe/`, `assets/mediapipe/`, `data/mediapipe/`.
