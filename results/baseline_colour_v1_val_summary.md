### Classical colour-masking (baseline-colour-v1) on val

**3D error is a proxy** (no physical ground truth): the distance between the 3D SURFACE POINT (not a corrected centre -- see src/geometry.py's scope note) computed from the predicted circle and from the labelled circle, through the same geometry module -- not distance to a physically measured apple, and these positions also carry a known ~3-4.5cm systematic bias toward the camera versus the true apple centre.

- Detection: precision=0.660, recall=0.404, F1=0.501, mAP@0.5=0.103 (TP=231, FP=119, FN=341)
- Overall (231 matched apples): median 2D error = 2.86px (0.08cm); median 3D proxy error = 0.10cm; within 2cm = 100.0%; within 3cm = 100.0%

| Distance band | n | median 2D err (cm) | median 3D proxy err (cm) | within 2cm | within 3cm |
|---|---|---|---|---|---|
| out_of_range | 231 | 0.08 | 0.10 | 100.0% | 100.0% |
