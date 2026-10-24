### Classical colour-masking (baseline-colour-v1) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.583, recall=0.593, F1=0.588, mAP@0.5=0.211 (TP=339, FP=242, FN=233)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 20 images): MAE=5.15 apples/image, RMSE=6.52, bias=+0.45 (over-counting on average), correlation=0.902
- Overall (339 matched apples): median 2D error = 2.75px (0.07cm); median 3D proxy error = 0.11cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 339 | 0.07 | 0.11 | 100.0% | 100.0% |
