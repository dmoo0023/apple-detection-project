# Stage 9 decision table (val split, identical evaluation code)

| Method | Precision | Recall | F1 | mAP@0.5 | Median 3D proxy (cm) | Within 2cm | Within 3cm | sec/image | Robustness |
|---|---|---|---|---|---|---|---|---|---|
| Classical (colour masking) | 0.537 | 0.561 | 0.548 | 0.204 | 0.11 | 100.0% | 100.0% | 0.020 | fails under shade (recall collapses); undersizes circles from leaf occlusion |
| YOLO (model-v1) | 0.785 | 0.842 | 0.812 | 0.830 | 0.07 | 100.0% | 100.0% | 0.050 | handles shade much better; dominant failure is missed detections on clustered/touching apples; occasional false positives on background structures |

**3D proxy numbers are distances between front-centre points and inherit the checkpoint-C1 depth-scale approximation — not literal centimetre accuracy against a physical apple, see docs/limitations.md.**
