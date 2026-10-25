### Classical colour-masking (baseline-colour-v1) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D front-centre point computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple.

- Detection: precision=0.537, recall=0.561, F1=0.548, mAP@0.5=0.204 (TP=425, FP=367, FN=333)
- **Apple count per image** (how close is the TOTAL count, not individual matches -- 26 images): MAE=4.92 apples/image, RMSE=6.26, bias=+1.31 (over-counting on average), correlation=0.915
- Overall (425 matched apples): median 2D error = 2.75px (0.07cm); median 3D proxy error = 0.11cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 425 | 0.07 | 0.11 | 100.0% | 100.0% |
