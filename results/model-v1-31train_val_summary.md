### YOLO yolov8n.pt (model-v1-31train) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.704, recall=0.837, F1=0.765, mAP@0.5=0.814 (TP=479, FP=201, FN=93)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 20 images): MAE=6.00 apples/image, RMSE=7.36, bias=+5.40 (over-counting on average), correlation=0.970
- Overall (479 matched apples): median 2D error = 1.89px (0.05cm); median 3D proxy error = 0.07cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 479 | 0.05 | 0.07 | 100.0% | 100.0% |
