### YOLO yolov8n.pt (model-v1-first-fit) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.676, recall=0.827, F1=0.744, mAP@0.5=0.756 (TP=473, FP=227, FN=99)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 20 images): MAE=7.00 apples/image, RMSE=8.32, bias=+6.40 (over-counting on average), correlation=0.968
- Overall (473 matched apples): median 2D error = 2.05px (0.06cm); median 3D proxy error = 0.07cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 473 | 0.06 | 0.07 | 100.0% | 100.0% |
