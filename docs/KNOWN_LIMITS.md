# Interpretation and release boundaries

- Fixed-ROI and reconstructed-track experiments have different inputs and evaluation populations. Do not pool their headline metrics.
- Original OT/GMM and all restoration strategies are retained, not selected after observing the best test F1. Hungarian and DoG remain competitive alternatives.
- ByteTrack uses a declared image-quality proxy, not native VLM confidence; OC-SORT uses a fixed pool without Byte low-score recovery.
- CNF_ROI_STAGE_ADAPTED is a custom ROI stage comparator; the Yang prior-9 K-means comparator is a separate adaptation, not a full reproduction of YOLO/ByteTrack/SAM.
- Synthetic checks verify code only. They are not experiment results and cannot establish generalization.
- Optical visible onset does not establish chemical ignition, detachment delay, physical height, particle size, intrinsic probability or a propagating front.
- Timing records concern the earlier implementation; corrected time advancement is explicitly versioned. Do not describe historical timings as a new benchmark.
- New-pressure performance and repeat-session human reliability remain unfinished until the respective evidence is obtained.
