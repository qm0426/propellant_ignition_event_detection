# Reproducibility and evidence

The authors confirmed historical timing used the same hardware and software environment: Intel Core i9-10980XE (18 cores/36 logical processors), about 256 GiB RAM, NVIDIA RTX 3090 24 GiB, driver 560.94. CPU/scientific and VLM stages use separate Python environments. Scientific versions: NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, scikit-learn 1.5.0, OpenCV 4.10.0.82; FPN torch 2.3.1+cu118 / torchvision 0.18.1+cu118. VLM torch 2.5.0+cu118 / transformers 4.57.4. This confirmation does not turn an earlier-code benchmark into a corrected-code benchmark.

SOURCE_MANIFEST.json records original and packaged hashes. Only legacy/default personal absolute paths were replaced where indicated; the active experiment sources were not edited. Third-party code is copied with license/provenance. Reproducing a historical source hash requires its original snapshot; the release's intentionally different path defaults will change hashes even when scientific rules are unchanged. Never copy an old completed status onto a new run.

The raw data, human judgments, pretrained model snapshot and trained FPN checkpoint must be supplied separately with permission. They are intentionally not in the source repository. A new-pressure run produces predictions without labels, but evaluating generalization still requires independent detection/identity/event reference data on that recording.

The included synthetic and unit checks establish software operability, not all quantitative paper claims. Real-cohort reruns require the full external data and original settings. No fresh full-cohort experiments are implied by packaging.
