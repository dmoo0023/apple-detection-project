### YOLO yolov8n.pt (model-v1-first-fit, NMS iou=0.3) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.754, recall=0.781, F1=0.767, mAP@0.5=0.744 (TP=447, FP=146, FN=125)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 20 images): MAE=3.25 apples/image, RMSE=4.10, bias=+1.05 (over-counting on average), correlation=0.966
- Overall (447 matched apples): median 2D error = 2.08px (0.06cm); median 3D proxy error = 0.08cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 447 | 0.06 | 0.08 | 100.0% | 100.0% |
