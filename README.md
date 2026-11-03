# Dense-particle optical event extraction — reviewer source snapshot

This repository contains the original-method pipeline and the review comparisons/ablations. It excludes recordings, confidential labels, model weights and generated scientific results. All paths below are relative to this checkout; run commands from the checkout root.

## Install and check

Use Python 3.10 or 3.11 in a separate environment. Install `requirements-core.txt`. For native trackers install the dependencies required by their preserved upstream files. For FPN install PyTorch 2.3.1/CUDA 11.8 with torchvision 0.18.1. For live LocateAnything use a **separate** environment with PyTorch 2.5.0/CUDA 11.8 and `requirements-detector.txt`; install the local model's own requirements as well. The dependency lists describe the known setup, not a complete transitive lock file.

```
python -m pip install -r requirements-core.txt
python -m unittest tests.revision.test_hmm_pressure tests.revision.test_log_detector tests.revision.test_mot_adaptations tests.revision.test_joint_validation
python -m tests.revision.make_synthetic_fixture --output outputs/synthetic_check
python -m ignition_revision judges --config outputs/synthetic_check/judges.json
```

Synthetic data are explicitly marked and must never be reported as real experimental validation.

## Original method on a new recording

Copy `revision_configs/pressure_pipeline.template.json` to a working config. Fill in image directory, physical-frame filename rule, recording ID, model directory and detector-environment Python executable. If multiple image formats contain duplicate frame numbers, select `image_extensions` (for example `[".bmp"]`) deliberately rather than discarding duplicates silently. Relative paths resolve from this checkout.

```
python run_pressure_pipeline.py --config revision_configs/pressure_pipeline.my_recording.json --dry-run
python run_pressure_pipeline.py --config revision_configs/pressure_pipeline.my_recording.json
python run_pressure_pipeline.py --config revision_configs/pressure_pipeline.my_recording.json --resume
```

The stages are open-vocabulary detection, box postprocessing, OT–Kalman tracking, Rescue/Gap/Prebirth support, ROI extraction and GMM–RTV event judgment. HMM is not inserted into this pipeline. Source/input fingerprints guard resume; changing source/config may require a new output directory. No GT is needed for prediction, but human reference data are required for performance claims.

## Independent and combined experiments

Fill the relevant `revision_configs/*.template.json` inputs. The templates intentionally require evidence/provenance and do not silently manufacture real GT or declare a protocol frozen. `run_submission_suite.py` selects run blocks; each block can contain one or multiple named methods/variants. Duplicate a block with a new ID and a one-method config to schedule any individual ablation separately.

```
python -m ignition_revision assignment --config revision_configs/assignment.my.json
python -m ignition_revision restoration --config revision_configs/restoration.my.json
python -m ignition_revision judges --config revision_configs/judges.my.json
python -m ignition_revision ablations --config revision_configs/ablations.my.json
python run_hmm_comparison.py --config revision_configs/judges.my.json --output outputs/hmm
python run_submission_suite.py --suite revision_configs/suite.my.json --only judges ablations
python run_submission_suite.py --suite revision_configs/suite.my.json --disable detection --resume
python run_submission_suite.py --suite revision_configs/suite.my.json
```

Change each suite block's config filename to your filled config before execution. `--dry-run` lists plans; it is not a data-validity or scientific-result check. The suite records exit codes and never treats a blocked run as completed.

## FPN train, inference and plotting

```
python build_ignition_dataset.py --help
python train_fpn_revision.py --config revision_configs/train_fpn_ignition.my.json
python infer_fpn_revision.py --config revision_configs/infer_fpn_ignition.my.json
python run_fpn_ignition_pipeline.py --config revision_configs/fpn_ignition_pipeline.my.json
python plot_revision_results.py --help
python plot_review_v2_results.py --help
```

Training/inference remain independent and are not accidentally triggered by the core suite. Select validation/test records explicitly in their dataset configuration. `plot_revision_results.py` reads individual family outputs; `plot_review_v2_results.py` reads historical review-v2 registry outputs, not the new minimal suite status file. Do not interchange these interfaces. Consult each command's `--help` for exact plotting arguments.

## Main source map

| Source | Experiment/function |
|---|---|
| test_locate_anything.py / postprocess_detection_txt_modified.py | VLM detection and original box cleaning |
| ignition_revision/detection.py | cached detection evaluation, cleaning ablation, LoG/MSER |
| ignition_revision/tracking.py, experiments.py | OT/Hungarian and R/G/P factorial experiments |
| ignition_revision/recovery.py | gap/prebirth support |
| ignition_revision/mot_adapters.py, mot_metrics.py | SORT/OC-SORT/quality-adapted ByteTrack and formal MOT metrics |
| ignition_revision/judges.py, hmm.py | GMM/HMM and decision integration |
| ignition_revision/paper_judges.py, prior9.py | SAS/DoG/stage and K-means prior-9 adaptations |
| ignition_revision/roi_io.py | original ROI features and physical-frame continuity |
| ignition_revision/experiments.py | decision-screen and feature ablations |
| ignition_revision/evaluation.py | identity-aware event matching and timing tolerance |
| ignition_revision/ignition_dataset.py, fpn_detector.py, fpn_events.py | full-frame single-class FPN dataset/train/infer and supplied-track event attribution |
| ignition_revision/pressure_pipeline.py | original method's combined new-recording pipeline |
| latest_experiment_scripts_20261018/ | historical 12-script scientific implementation used by compatibility loaders |
| prompt_sensitivity_tools/ | prompt sensitivity experiment and evaluation |

See `docs/INPUTS_OUTPUTS.md`, `docs/REPRODUCIBILITY.md`, `docs/KNOWN_LIMITS.md`, `NOTICE.md` and `SOURCE_MANIFEST.json`. This package is locally prepared for review; it has not been published to a remote repository.
