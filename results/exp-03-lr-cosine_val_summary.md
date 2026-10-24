### YOLO yolov8n.pt (exp-03-lr-cosine) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.765, recall=0.839, F1=0.801, mAP@0.5=0.824 (TP=636, FP=195, FN=122)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=3.42 apples/image, RMSE=4.59, bias=+2.81 (over-counting on average), correlation=0.981
- Overall (636 matched apples): median 2D error = 1.79px (0.05cm); median 3D proxy error = 0.07cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 636 | 0.05 | 0.07 | 100.0% | 100.0% |
