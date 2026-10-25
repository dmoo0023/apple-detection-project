# Report figures and tables

Collected by `scripts/collect_report_assets.py` from the pipeline's outputs. All 3D numbers inherit the approximate depth scale and camera intrinsics (see `docs/limitations.md`), and 3D error is a proxy (no physical ground truth).


## 01_data

- `01_data/R03U02_overlay.png` — Depth image over the RGB image, checking that depth is aligned to colour (Stage 1).
- `01_data/R15U07_overlay.png` — Depth image over the RGB image, checking that depth is aligned to colour (Stage 1).
- `01_data/R01U02_Summer_RGB-12-40-26_labels.png` — Labelled apples: labelme circle (red), derived YOLO box (green), centre (blue) (Stage 2).
- `01_data/R01U03_Summer_RGB-12-40-37_labels.png` — Labelled apples: labelme circle (red), derived YOLO box (green), centre (blue) (Stage 2).

## 02_classical_baseline

- `02_classical_baseline/R01U01_Summer_RGB-12-40-11_baseline.png` — Classical colour-masking detections on a val image (Stage 6).
- `02_classical_baseline/R01U04_Summer_RGB-12-41-9_baseline.png` — Classical detections on a denser val image (Stage 6).
- `02_classical_baseline/stage6_grid_search.csv` — Threshold grid search on train: every config tried and its F1.
- `02_classical_baseline/baseline_colour_v1_val_summary.md` — Classical baseline val results (detection, count and error tables).

## 03_yolo_training

- `03_yolo_training/results.png` — Training/validation loss and metric curves over 100 epochs for the chosen model.
- `03_yolo_training/BoxPR_curve.png` — Precision-recall curve of the chosen model (Ultralytics validation).
- `03_yolo_training/BoxF1_curve.png` — F1 against confidence threshold for the chosen model.
- `03_yolo_training/confusion_matrix_normalized.png` — Normalised confusion matrix (apple vs background).
- `03_yolo_training/val_batch0_pred.jpg` — Ultralytics val batch: predictions with confidences.
- `03_yolo_training/val_batch0_labels.jpg` — The same val batch with ground-truth labels, for comparison.
- `03_yolo_training/R01U01_Summer_RGB-12-40-11_yolo.png` — Chosen YOLO model's detections (green) vs labels (red) on a val image.
- `03_yolo_training/R01U04_Summer_RGB-12-41-9_yolo.png` — Chosen YOLO model's detections vs labels on a denser val image.
- `03_yolo_training/learning_curve.png` — Learning curve: F1/mAP and count error against number of labelled train images.
- `03_yolo_training/learning_curve.csv` — Learning-curve numbers (all points on the same 26-image val set).
- `03_yolo_training/stage8_comparison.csv` — Stage 8 experiments: settings actually used and val metrics for every run.
- `03_yolo_training/stage8_seed_summary.csv` — Seed repeats: mean and spread of F1, mAP and count error.

## 04_error_analysis

- `04_error_analysis/R01U04_Summer_RGB-12-41-9_errors.png` — Chosen model's errors on a val image: labelled apples (thin green), missed apples (red), false positives (yellow).
- `04_error_analysis/R08U01_Summer_RGB-13-25-43_errors.png` — Error overlay on a second val image.
- `04_error_analysis/R09U03_Summer_RGB-13-31-23_errors.png` — Error overlay on a third val image.
- `04_error_analysis/errors.csv` — Every false negative and false positive of the chosen model on val.

## 05_method_comparison

- `05_method_comparison/stage9_decision_table.md` — Classical vs YOLO on val with identical evaluation (Stage 9 decision table).
- `05_method_comparison/stage9_decision_table.csv` — Decision table as CSV.

## 06_final_test

- `06_final_test/final_test_eval_summary.md` — One-time locked test evaluation of the chosen model.

## 07_3d_output

- `07_3d_output/R01U01_3d_overlay.png` — R01U01: photo with numbered apples beside the 3D graph of their front-centre points.
- `07_3d_output/R08U01_3d_overlay.png` — R08U01: photo with numbered apples beside the 3D graph of their front-centre points.
- `07_3d_output/R23U14_3d_overlay.png` — R23U14: photo with numbered apples beside the 3D graph of their front-centre points.
- `07_3d_output/R04U13_3d_overlay.png` — R04U13: photo with numbered apples beside the 3D graph of their front-centre points.
- `07_3d_output/R10U14_3d_overlay.png` — R10U14: photo with numbered apples beside the 3D graph of their front-centre points.
- `07_3d_output/R01U01_3d_overlay.pdf` — R01U01: same figure as vector PDF.
- `07_3d_output/R08U01_3d_overlay.pdf` — R08U01: same figure as vector PDF.
- `07_3d_output/R23U14_3d_overlay.pdf` — R23U14: same figure as vector PDF.
- `07_3d_output/R04U13_3d_overlay.pdf` — R04U13: same figure as vector PDF.
- `07_3d_output/R10U14_3d_overlay.pdf` — R10U14: same figure as vector PDF.
- `07_3d_output/R01U01_3d.png` — R01U01: 3D graph of the front-centre points on its own.
- `07_3d_output/R08U01_3d.png` — R08U01: 3D graph of the front-centre points on its own.
- `07_3d_output/R23U14_3d.png` — R23U14: 3D graph of the front-centre points on its own.
- `07_3d_output/R04U13_3d.png` — R04U13: 3D graph of the front-centre points on its own.
- `07_3d_output/R10U14_3d.png` — R10U14: 3D graph of the front-centre points on its own.
- `07_3d_output/stage10_histograms.png` — All 544 trees: apples per tree and depth (Z) of every detected apple.
- `07_3d_output/stage10_summary.md` — Summary of the full 544-tree run (counts, missing depth, depth range).
- `07_3d_output/stage-4-worked-example.md` — Worked example: one apple's 3D front-centre point calculated by hand.

## Tables built for the report

- `01_data/data_split.csv` — Train/val/test split: orchard rows, trees, labelled images and apples per split.
- `06_final_test/val_vs_test.csv` — Chosen model on val vs the locked test set.
- `04_error_analysis/error_breakdown.csv` — Error counts by type, with mean size and confidence.
