# Stage 3: Splits

## What this stage does

Before any model is trained, the 544 trees need to be permanently divided
into **train** (used to fit the model), **validation/val** (used to compare
settings and pick the best one), and **test** (touched exactly once, at the
very end, to report a final honest number). This has to happen *before*
any modelling starts and then **never change** — if you peek at test-set
results while still making decisions, the test set quietly stops being a
fair measure of how the model does on new data.

## Why whole rows, not random individual trees

The 544 trees come from 23 physical orchard rows (`R01`–`R23`), each with
anywhere from 9 to 47 trees. Trees in the same row were photographed close
together in time and space, so they likely share lighting, background
plants, trellis wires, and camera angle. If one tree from row 7 is in
training and the tree right next to it is in test, the model could do well
on that test tree partly by recognising the *background*, not the apples —
an inflated, misleading test score. Assigning whole rows to a single split
avoids this "leakage."

## What was done

`scripts/make_splits.py`:

1. Grouped the manifest's 544 trees by row (23 rows).
2. Shuffled the 23 rows with a **fixed seed (42)**, then walked through them
   in that shuffled order, filling train until ~70% of all trees were
   assigned, then val until ~85%, with everything left over going to test.
   Because rows vary in size (9–47 trees), the result won't be an exact
   70.0/15.0/15.0 split — it came out **70.4% / 15.4% / 14.2%**, close
   enough.
3. Wrote `configs/splits.json`: which row went to which split, plus the
   full list of tree IDs per split, the seed, and the method used — enough
   to explain or regenerate the exact same split to an examiner.
4. The script **refuses to overwrite** an existing `splits.json` unless run
   with `--overwrite`, so a split can't be silently changed later (rule 6
   in Section 3: commit `splits.json` once and never change it).
5. Reported labelled-image counts per split and a labelling recommendation
   (see below).

## Result

| Split | Trees | % | Labelled now |
|---|---|---|---|
| train | 383 | 70.4% | 5 |
| val | 84 | 15.4% | 20 |
| test | 77 | 14.2% | **0** |

By chance of the fixed seed, row 1 (all 20 of its images already labelled)
landed in **val**, and row 2 (5 labelled) landed in **train**. No currently
labelled image is in **test**. This is an honest consequence of random row
assignment with only 2 labelled rows so far — it is reported rather than
engineered away, per rule 9 (never weaken a check, and don't quietly
re-roll the seed until the numbers look more convenient).

**Practical effect:** Stage 5/6 work can validate against the 20 labelled
val images, but the test set is not yet usable for anything (which is
fine — rule 3 says it must stay untouched until Stage 9 anyway). To reach
the brief's target of 50–80 labelled test images, `scripts/make_splits.py`
prints a specific list of 50 unlabelled test-split tree IDs to label next
(from rows 4 and 21) — see `data/derived/stage3_report.md` for the full
list. This feeds directly into Stage 7's pre-labelling loop.

## What could go wrong later

- If `configs/splits.json` is ever deleted and regenerated, the seed
  reproduces the *same* row shuffle — but only if the manifest's row list
  hasn't changed (e.g. if more trees were somehow added). Re-running
  without `--overwrite` is the safety net against doing this by accident.
- The recommended "next 50 test images to label" are just the first 50
  test-split trees in ID order, not chosen for diversity (e.g. apple
  density, lighting). If the student wants a more representative test
  sample, they should browse the recommended rows and pick visually
  varied trees rather than labelling strictly in order.

## Terms used

- **Train/val/test split:** three non-overlapping subsets of the data —
  train fits the model, val is used for comparisons during development,
  test is the untouched final check.
- **Leakage:** when information that shouldn't be available at evaluation
  time (e.g. a shared background between a train and a test image)
  artificially inflates how good a model looks.
