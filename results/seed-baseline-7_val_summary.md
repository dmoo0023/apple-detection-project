### YOLO yolov8n.pt (seed-baseline-7) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.779, recall=0.831, F1=0.804, mAP@0.5=0.832 (TP=630, FP=179, FN=128)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=3.12 apples/image, RMSE=4.36, bias=+1.96 (over-counting on average), correlation=0.975
- Overall (630 matched apples): median 2D error = 1.78px (0.05cm); median 3D proxy error = 0.07cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 630 | 0.05 | 0.07 | 100.0% | 100.0% |
