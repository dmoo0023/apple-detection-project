### Classical colour-masking (baseline-colour-v1) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.548, recall=0.572, F1=0.560, mAP@0.5=0.200 (TP=403, FP=332, FN=301)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 25 images): MAE=5.00 apples/image, RMSE=6.36, bias=+1.24 (over-counting on average), correlation=0.905
- Overall (403 matched apples): median 2D error = 2.77px (0.08cm); median 3D proxy error = 0.11cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 403 | 0.08 | 0.11 | 100.0% | 100.0% |
