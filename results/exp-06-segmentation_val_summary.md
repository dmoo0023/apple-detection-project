### YOLO-seg yolov8n-seg.pt (exp-06-segmentation) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.686, recall=0.665, F1=0.675, mAP@0.5=0.654 (TP=504, FP=231, FN=254)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=2.12 apples/image, RMSE=2.81, bias=-0.88 (under-counting on average), correlation=0.984
- Overall (504 matched apples): median 2D error = 3.16px (0.09cm); median 3D proxy error = 0.12cm; within 2cm = 99.8%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 504 | 0.09 | 0.12 | 99.8% | 100.0% |
