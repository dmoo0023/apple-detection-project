### YOLO yolov8n.pt (seed-baseline-123) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.818, recall=0.810, F1=0.814, mAP@0.5=0.801 (TP=614, FP=137, FN=144)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=2.27 apples/image, RMSE=3.09, bias=-0.27 (under-counting on average), correlation=0.981
- Overall (614 matched apples): median 2D error = 1.76px (0.05cm); median 3D proxy error = 0.07cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 614 | 0.05 | 0.07 | 100.0% | 100.0% |
