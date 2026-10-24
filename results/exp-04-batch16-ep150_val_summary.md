### YOLO yolov8n.pt (exp-04-batch16-ep150) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.834, recall=0.796, F1=0.814, mAP@0.5=0.801 (TP=603, FP=120, FN=155)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=2.73 apples/image, RMSE=3.50, bias=-1.35 (under-counting on average), correlation=0.977
- Overall (603 matched apples): median 2D error = 1.71px (0.05cm); median 3D proxy error = 0.06cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 603 | 0.05 | 0.06 | 100.0% | 100.0% |
