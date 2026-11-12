# Trajectory-resolved optical-event detection

Method implementation for multi-particle imaging:

**LocateAnything → candidate cleaning → OT–Kalman association with Rescue, Gap and Prebirth → ROI features → GMM–RTV event judgment.**

[中文完整运行说明](README_中文.md)

This repository contains the method only. It does not require reference labels and does not include comparison algorithms, ablation runners, FPN training/inference, annotation websites, paper plotting or evaluation metrics. Raw images and model weights are supplied separately. Outputs describe particle-associated visible optical transitions; onset positions are image-plane trajectory ROI locations.

## Setup

Python 3.10 is recommended. Install the CPU pipeline dependencies:

```console
python -m pip install -r requirements-core.txt
```

Live detection additionally requires the existing LocateAnything-compatible PyTorch/Transformers environment and a local LocateAnything-3B model snapshot. Additional dependencies are listed in `requirements-detector.txt`; install PyTorch appropriate to your CUDA environment separately. Cached-box processing does not import Torch or Transformers. Set `detector.python` to use a separate detection environment, or leave it `null` for the current interpreter.

## Inputs and execution

1. Supply consecutive full-resolution 8-bit RGB, RGBA or grayscale frames of identical dimensions. Set the filename regex and image extension in the configuration. The default regex extracts the trailing `_00000000` frame number. Internal frames use zero-based global recording indices.
2. Copy and edit `configs/method.template.json`: set `image_dir`, `output_dir`, `recording_id`, `image_extensions` and `detector.model`.
3. Run:

```console
python run_method.py --config configs/method.template.json
```

All relative paths are resolved against the repository root. `run_pressure_pipeline.py` is an equivalent entry point.

To reuse saved detections, edit `configs/method.cached_txt.json` and run:

```console
python run_method.py --config configs/method.cached_txt.json
```

Provide one TXT per image with the same stem. Each line is `label x1 y1 x2 y2` in original-image pixel coordinates. Empty files represent no detections; missing files are errors. A sixth column is accepted but not interpreted as native detection confidence. This mode imports boxes at the `detect` stage and then runs the remaining stages normally.

## Stages and resuming

```console
python run_method.py --config configs/method.template.json --dry-run
python run_method.py --config configs/method.template.json --stop-after detect
python run_method.py --config configs/method.template.json --start-from clean --resume
python run_method.py --config configs/method.template.json --resume
```

Stages are `detect`, `clean`, `track`, `judge`. Earlier stages must already be complete to use `--start-from`. Per-stage checksums and per-frame detector checkpoints protect resuming. Changing inputs, configuration or code requires a new output directory. Use a fresh directory for this packaged version rather than reusing the source-hash lock of the original project.

The templates retain the working nominal configuration, including all three restoration strategies, seed 2026, incremental Kalman state timestamps, contiguous-frame difference features and the first-terminal GMM candidate with OR-RTV. Resolved default parameters are in `docs/nominal_parameters.json` and each run's effective configuration files. Feature normalization uses the entire trajectory; this is an offline method.

## Outputs

| Path under `output_dir` | Contents |
|---|---|
| `frames.csv`, `pipeline_config.json`, `pipeline_status.json` | Input manifest, fingerprints and stage status |
| `source_snapshot/` | Core source snapshot used in the run |
| `01_detection/raw_candidates.csv` | Raw candidates; live mode also writes `per_frame/` JSON/TXT |
| `02_cleaning/candidates.csv`, `cleaning_audit.csv` | Retained boxes and cleaning provenance |
| `03_tracking/track_observations.csv`, `association_trace.csv` | Observations, identities, observation sources and association trace |
| `03_tracking/roi_manifest.csv` | Original images, crop bounds and global ROI positions |
| `tracks.csv` | Compatibility table using the `frame_id` column |
| `03_tracking/motchallenge/` | Format export only, with global frame +1; no identity metrics |
| `04_judgment/events.csv` | All tracks, including negative judgments |
| `04_judgment/GMM_RTV/` | Predictions, candidate audit, RTV channels and diagnostic traces |
| `runtime_summary.csv` | Stage and nested operation timings; overlapping totals must not be summed |

Filter `events.csv` by `has_ignition=True` for predicted events. `onset_frame` and `confirmation_frame` denote the estimated optical onset and subsequent confirmation frame. `onset_cx/cy` and `onset_x1/y1/x2/y2` refer to the original image coordinates of the supporting trajectory ROI. Restored positions are not independently measured particle contours. Sources distinguish `detector_primary`, `detector_rescue`, `gap_roi` and `prebirth_roi`. `native_score` remains empty when no native detector confidence is available.

## Implementation map

`method_core/` contains the selected reachable definitions of the four working method modules: detector, box cleaning, OT–Kalman/restoration and ROI/GMM features. `ignition_revision/` provides frame validation, orchestration, ROI reading, nominal GMM–RTV judgment and result exports. `SOURCE_MANIFEST.json` records the original source hashes and packaging changes. Algorithm alternatives and experiment orchestration have been removed.

## Verification

```console
python -m unittest discover -s tests -v
```

Generated fixtures check the cached-box pipeline, empty inputs, frame validation, incremental filter time, raw-feature RTV, coordinate exports and checksum-protected resuming. A one-time parity check against the working implementation confirmed identical cleaning, tracking including a gap, features, GMM candidates, RTV and judgments on the same fixture. These checks are software verification, not scientific experiment results. The live GPU model was not rerun during packaging; its selected source definitions and prompts were retained. See `VERIFICATION.json`.

See `NOTICE.md` for license status and third-party model notices.
