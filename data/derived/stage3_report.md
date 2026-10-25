# Stage 3 split report

Wrote configs\splits.json (seed=42).

## Row assignment
- train: rows [2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23]
- val: rows [1, 8, 9, 19]
- test: rows [4, 21]

## Counts per split
- **train**: 383 trees (70.4%), 7 labelled
- **val**: 84 trees (15.4%), 20 labelled
- **test**: 77 trees (14.2%), 0 labelled

## Labelling recommendation (target: 50-80 labelled test images)
- test split currently has only 0 labelled images. Recommend labelling at least 50 more from the test split to reach the low end of the target (50).
- suggested trees to label next (from test split, unlabelled, in tree-ID order):
  ['R04U01', 'R04U02', 'R04U03', 'R04U04', 'R04U05', 'R04U06', 'R04U07', 'R04U08', 'R04U09', 'R04U10', 'R04U11', 'R04U12', 'R04U13', 'R04U14', 'R04U15', 'R04U16', 'R04U17', 'R04U18', 'R04U19', 'R04U20', 'R04U21', 'R04U22', 'R04U23', 'R04U24', 'R04U25', 'R04U26', 'R04U27', 'R04U28', 'R04U29', 'R04U30', 'R04U31', 'R21U01', 'R21U02', 'R21U03', 'R21U04', 'R21U05', 'R21U06', 'R21U07', 'R21U08', 'R21U09', 'R21U10', 'R21U11', 'R21U12', 'R21U13', 'R21U14', 'R21U15', 'R21U16', 'R21U17', 'R21U18', 'R21U19']
- note: these are **test** images — per rule 3, do not look at their predictions or metrics until the Stage 9 final evaluation, only label them.

- val split has 20 labelled images. Stage 5/6 need at least a handful of labelled val images to compute meaningful metrics; if this is 0, prioritise labelling some val-split images too.
