### YOLO yolov8n.pt (model-v1-first-fit) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D front-centre point computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple.

- Detection: precision=0.736, recall=0.772, F1=0.753, mAP@0.5=0.742 (TP=585, FP=210, FN=173)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=3.19 apples/image, RMSE=4.19, bias=+1.42 (over-counting on average), correlation=0.969
- Overall (585 matched apples): median 2D error = 2.02px (0.06cm); median 3D proxy error = 0.08cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 585 | 0.06 | 0.08 | 100.0% | 100.0% |
