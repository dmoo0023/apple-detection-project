### YOLO (model-v1) -- FINAL TEST EVALUATION on test

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.753, recall=0.761, F1=0.757, mAP@0.5=0.755 (TP=644, FP=211, FN=202)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 20 images): MAE=4.55 apples/image, RMSE=5.34, bias=+0.45 (over-counting on average), correlation=0.933
- Overall (644 matched apples): median 2D error = 2.20px (0.06cm); median 3D proxy error = 0.09cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 643 | 0.06 | 0.09 | 100.0% | 100.0% |
| unknown | 1 | nan | nan | nan% | nan% |
