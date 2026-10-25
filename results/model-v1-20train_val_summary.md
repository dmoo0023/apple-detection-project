### YOLO yolov8n.pt (model-v1-20train) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D front-centre point computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple.

- Detection: precision=0.773, recall=0.790, F1=0.781, mAP@0.5=0.779 (TP=599, FP=176, FN=159)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=2.73 apples/image, RMSE=3.46, bias=+0.65 (over-counting on average), correlation=0.977
- Overall (599 matched apples): median 2D error = 1.81px (0.05cm); median 3D proxy error = 0.07cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 599 | 0.05 | 0.07 | 100.0% | 100.0% |
