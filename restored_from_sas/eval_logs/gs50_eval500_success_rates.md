# Global Step 50 Evaluation Results

Both suites were evaluated on 500 trajectories, with 50 trajectories per task. Success means `success_once` during the episode.

## Summary

| Task suite | Successes | Trajectories | Overall success rate |
|---|---:|---:|---:|
| LIBERO-10 | 209 | 500 | 41.8% |
| LIBERO-Spatial | 425 | 500 | 85.0% |
| Combined | 634 | 1000 | 63.4% |

## LIBERO-10

**Overall: 209/500 (41.8%)**

| ID | Task | Successes | Trials | Success rate |
|---:|---|---:|---:|---:|
| 0 | Put both the alphabet soup and the tomato sauce in the basket | 21 | 50 | 42.0% |
| 1 | Put both the cream cheese box and the butter in the basket | 1 | 50 | 2.0% |
| 2 | Turn on the stove and put the moka pot on it | 14 | 50 | 28.0% |
| 3 | Put the black bowl in the bottom drawer of the cabinet and close it | 45 | 50 | 90.0% |
| 4 | Put the white mug on the left plate and put the yellow and white mug on the right plate | 34 | 50 | 68.0% |
| 5 | Pick up the book and place it in the back compartment of the caddy | 47 | 50 | 94.0% |
| 6 | Put the white mug on the plate and put the chocolate pudding to the right of the plate | 14 | 50 | 28.0% |
| 7 | Put both the alphabet soup and the cream cheese box in the basket | 22 | 50 | 44.0% |
| 8 | Put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | Put the yellow and white mug in the microwave and close it | 11 | 50 | 22.0% |

## LIBERO-Spatial

**Overall: 425/500 (85.0%)**

| ID | Task | Successes | Trials | Success rate |
|---:|---|---:|---:|---:|
| 0 | Pick up the black bowl between the plate and the ramekin and place it on the plate | 49 | 50 | 98.0% |
| 1 | Pick up the black bowl next to the ramekin and place it on the plate | 48 | 50 | 96.0% |
| 2 | Pick up the black bowl from table center and place it on the plate | 50 | 50 | 100.0% |
| 3 | Pick up the black bowl on the cookie box and place it on the plate | 47 | 50 | 94.0% |
| 4 | Pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 44 | 50 | 88.0% |
| 5 | Pick up the black bowl on the ramekin and place it on the plate | 19 | 50 | 38.0% |
| 6 | Pick up the black bowl next to the cookie box and place it on the plate | 48 | 50 | 96.0% |
| 7 | Pick up the black bowl on the stove and place it on the plate | 50 | 50 | 100.0% |
| 8 | Pick up the black bowl next to the plate and place it on the plate | 49 | 50 | 98.0% |
| 9 | Pick up the black bowl on the wooden cabinet and place it on the plate | 21 | 50 | 42.0% |

## Source reports

- `results/libero10_gs50_eval500/20260828_145916/openpi05_libero_10_binary_gs50_fixed500/eval_report.json`
- `results/spatial_gs50_eval500_5gpu/20260828_154050/openpi05_libero_spatial_binary_gs50_fixed500/eval_report.json`
