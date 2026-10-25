# Stage 10 summary: full inference over all 544 trees

- Total detections: 19294
- Trees processed: 544 (all 544 in the manifest, including 1 with zero detections, counted explicitly rather than silently dropped from this summary)
- Apples per tree: min=0, median=36, mean=35.47, max=94
- Detections with no usable depth sample: 141 (0.7%)
- Z (depth) of valid detections: min=0.031m, median=0.144m, max=0.179m
- **100.0% of valid Z values fall below the dataset's own stated 0.4m minimum capture distance** -- the checkpoint-C1 depth-scale limitation propagated through the full 544-tree run exactly as expected, not a new problem. See docs/limitations.md.
