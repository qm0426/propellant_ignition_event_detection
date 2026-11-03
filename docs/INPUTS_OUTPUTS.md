# Data interfaces

The JSON schemas under revision_configs/schemas and each module's validation are authoritative. Do not substitute a predicted track ID for a manually reviewed reference identity.

| File | Required role / key fields |
|---|---|
| frames.csv | recording_id, frame_idx (global zero-based physical frame), image_path; annotated flags where applicable |
| candidates.csv | recording_id, frame_idx, candidate_id, x1,y1,x2,y2, prompt_group; native_score is empty when unavailable; score_semantics declares quality proxies |
| reference_boxes.csv | recording_id, frame_idx, ref_object_id, x1,y1,x2,y2; reviewed identities and ignore handling |
| reference_events.csv | recording_id, ref_object_id, ref_event_id, onset_frame |
| rois.csv | recording_id, track_id, frame_idx, roi_path; explicit physical frame mapping |
| labels.csv | recording_id, track_id, ref_object_id, has_ignition, onset_frame; all negatives included explicitly |

Coordinates are pixel image coordinates; y increases downward. Box geometry must have positive area. Do not assume segmentation or calibrated physical size is available. Keep track/recording keys unique and missing frames explicit. Input image paths inside CSV must be updated after moving a dataset; repository relocation cannot relocate your separate raw data automatically.

Historical inputs can be adapted through `ignition_revision/legacy_inputs.py`, using the config's legacy input section rather than changing scientific code. Preserve original tracks.csv, labels and their physical-frame map. Labels alone cannot replace per-frame identity boxes for HOTA/IDF1/IDSW.

Per-family results are fingerprinted subdirectories below configured output_dir: resolved_config.json, run metadata and status, method-level predictions/events/tracks and evaluation CSV/JSON. Inspect run status before interpreting a table. Missing/partial/blocked jobs are not valid completed measurements. Exact files vary by family; the synthetic check is a small inspectable working example.

Full-frame FPN uses single-class ignition foreground boxes with explicit supervised/partial-label masks. Existing supplied tracks attribute predictions to particles; this is not unconstrained end-to-end particle identity recovery. Unlabeled instances must not silently become background. Preserve recording-level splits and do not report training data as test performance.
