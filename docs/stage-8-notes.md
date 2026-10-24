# Stage 8: Training and experiments (in progress — first model only)

## Status

This note covers Stage 8 item 1 only ("first model"), done now at the
student's request, ahead of the rest of Stage 8, because more labels are
still being added and the plan is to retrain as they arrive (see
`EXPERIMENTS.md`'s `model-v1-first-fit` entry for the honest caveats on
reading too much into this one run). The full Stage 8 series (item 3: one
change at a time across model size, pretrained vs. random init, learning
rate/batch/epochs, augmentation, boxes vs. segmentation), seed repeats
(item 4), the learning-curve plot (item 5), and error analysis (item 6)
are not done yet — they need more than one labelled-image-count data point
to mean anything, which is exactly what's coming as more labels land.

## What was built

1. **`scripts/prepare_yolo_dataset.py`** — Ultralytics expects images and
   labels in parallel `images/<split>/` and `labels/<split>/` folders (it
   finds one from the other by swapping `/images/` for `/labels/` in the
   path). Our raw images live in OneDrive and converted labels in
   `data/derived/labels_det/` (flat), so this script copies the currently
   labelled train/val images+labels into that expected layout under
   `data/derived/yolo_dataset/` — small enough to copy rather than symlink
   (simpler and more reliable on Windows), and git-ignored since it's
   fully regenerable from raw data + existing label files.
2. **`scripts/train_yolo.py`** — fine-tunes a pretrained YOLO model
   (`yolov8n.pt` by default) with `workers=0` (the brief's own note:
   Ultralytics' data loader can hang/crash on Windows otherwise), then runs
   the trained model on val and evaluates it through the **same**
   `src/evaluate.py` used for the classical baseline — so the two are
   directly comparable on identical metrics, not each reporting its own
   favourite number.
3. **YOLO box → apple circle conversion:** a YOLO detection is a box
   `(x0, y0, x1, y1)`; this project's geometry/evaluation code works with
   circles `(cx, cy, radius)`. Converted as centre = box centre, radius =
   average of half-width and half-height — a reasonable approximation
   since most detected boxes are close to square (apples are round), not
   an exact statement that every box is a perfect square.

## Result

Classical's numbers below are after its own re-tune on the now-10-image
train split (see `docs/stage-6-notes.md`'s update note) — both methods
compared on the same, current train/val split.

| | Classical baseline | YOLO (first fit) |
|---|---|---|
| Precision | 0.583 | 0.676 |
| Recall | 0.593 | 0.827 |
| F1 | 0.588 | **0.744** |
| mAP@0.5 | 0.211 | **0.756** |
| Count MAE (apples/image) | **5.15** | 7.00 |
| Count bias | **+0.45** | +6.40 |
| Count correlation | 0.902 | **0.968** |

Full tables: `results/model-v1-first-fit_val_summary.md` /
`..._val_matches.csv`. YOLO wins clearly on every per-apple metric
(F1, mAP, precision, recall) — see the caveats in `EXPERIMENTS.md` before
reading this as a finished result (small training set, and train/val being
adjacent orchard rows means some of this apparent strength could be domain
similarity rather than a method that will hold up as more, visually
different rows get labelled).

**But classical wins on apple COUNT accuracy** (added as a metric after a
student request): MAE 5.15 vs YOLO's 7.00, and a far smaller over-counting
bias (+0.45 vs +6.40). This is not a contradiction — it's exactly what the
two metrics are built to show differently. YOLO's much higher recall
(0.827 vs 0.593) comes from making substantially more total predictions
(700 vs 581 across 20 val images, for 572 true apples), and while a higher
fraction of YOLO's predictions are individually correct (precision 0.676
vs 0.583), the raw volume of extra predictions pushes its per-image counts
further from the true count on average than classical's more conservative
(lower-recall, lower-volume) predictions do. **Which metric matters more
depends on the application**: a picking robot cares about each apple's
individual position (F1/mAP territory), while a yield-estimation use case
cares more about getting the total count right (this new metric's
territory) — this is a genuine, non-obvious trade-off worth keeping in the
final report, not a bug in either number.

## A training-curve wobble worth noting

`runs/yolo/model-v1-first-fit/results.png` shows a sharp transient spike
around epoch ~30 (val/cls_loss briefly exceeds 100; precision/recall/mAP
dip at the same point), before recovering and climbing steadily to the
final values. Most likely explanation: batch size 4 with only 20
validation images means very few validation batches per epoch, so one
unusual batch can swing the aggregate loss a lot. `best.pt` (used for the
results above) is selected by tracking fitness across all epochs, and
training clearly recovers and keeps improving afterward — not treated as a
problem, but flagged here rather than left as an unexplained blip in a
plot nobody commented on. Worth watching whether it recurs, shrinks, or
disappears once retraining happens with more labelled images (bigger,
more stable batches).

## Why YOLO's own printed metrics differ slightly from ours

During training, Ultralytics prints its own mAP50 (0.805) — close to, but
not identical to, this project's mAP@0.5 (0.756). Expected, not a bug: the
two use slightly different box representations (Ultralytics' native boxes
vs. the radius-from-average-half-width/height circles this project
converts to) and Ultralytics computes metrics on all 20 val images in one
pass where this project's number comes from the same evaluation code every
other method is scored with. When reporting a method's score anywhere in
this project, use **this project's own evaluate.py output**, not a
training framework's own printed number, so every method is held to the
exact same measuring stick (rule 7).

## What could go wrong later

- Retraining as more labels arrive (the planned next step) will add real
  points to a learning curve — watch specifically whether F1/mAP keep
  climbing, plateau, or (if train and val stay dominated by just 1-2 rows
  each) stay suspiciously flat in a way that suggests the model is mostly
  learning this orchard's specific rows rather than apples in general.
- `runs/yolo/<run_id>/weights/*.pt` are git-ignored (rule: model weights
  aren't committed) — only the config/metrics in `runs/experiments.csv`
  and `results/` are kept in git. If a specific run's weights need to be
  kept long-term, that's a question for the student (Git LFS / DVC / a
  separate weights store), not something to assume.

## Terms used

- **Fine-tuning:** continuing to train a model that's already been trained
  on a large, different dataset (here, YOLO's standard pretrained weights),
  rather than starting from random weights — lets a small, task-specific
  dataset still produce a useful model, since the pretrained weights
  already know general shapes/edges/colours.
- **Learning curve:** a plot of model performance against how much labelled
  training data was used — shows whether more labelling effort is still
  worth it, or whether performance has levelled off.
