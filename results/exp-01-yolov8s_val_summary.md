### YOLO yolov8s.pt (exp-01-yolov8s) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.789, recall=0.839, F1=0.813, mAP@0.5=0.826 (TP=636, FP=170, FN=122)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=2.85 apples/image, RMSE=4.42, bias=+1.85 (over-counting on average), correlation=0.971
- Overall (636 matched apples): median 2D error = 1.76px (0.05cm); median 3D proxy error = 0.06cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 636 | 0.05 | 0.06 | 100.0% | 100.0% |
