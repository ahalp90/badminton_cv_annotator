# Exact dense-response feature reuse sketch

Current hot expression in `continuous_support`:

```python
distance = maps[family, pixel_y, pixel_x]
response = np.exp(
    -0.5 * np.square(distance / assignment.DISTANCE_SIGMA_PX)
).mean(axis=2)
```

Candidate rewrite for sufficiently large pairs:

```python
response_maps = np.exp(
    -0.5 * np.square(maps / assignment.DISTANCE_SIGMA_PX)
)
response = response_maps[family, pixel_y, pixel_x].mean(axis=2)
```

Conditions:

- keep the exact dtype used by the baseline;
- do not regroup the mean or later marking reductions;
- use a measured court-count threshold, because dense precomputation loses on small pairs;
- compare score bytes on saved real pair inputs under the Carmack NumPy build;
- rerun all 28 views and require exact saved results.

The pack's synthetic benchmark found a break-even between 1,024 and 4,096 courts and 1.15× speed-up for the post-projection kernel at 16,384 courts. The whole detector gain will be smaller.
