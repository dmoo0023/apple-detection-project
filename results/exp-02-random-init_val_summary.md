### YOLO yolov8n.yaml (exp-02-random-init) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.737, recall=0.711, F1=0.724, mAP@0.5=0.656 (TP=539, FP=192, FN=219)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=4.04 apples/image, RMSE=5.30, bias=-1.04 (under-counting on average), correlation=0.955
- Overall (539 matched apples): median 2D error = 2.42px (0.07cm); median 3D proxy error = 0.10cm; within 2cm = 99.8%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 539 | 0.07 | 0.10 | 99.8% | 100.0% |
