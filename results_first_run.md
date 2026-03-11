# First Run Results

## Setup

- Data window: `2000-2008`
- Physics window: `2000-2008`
- Steps: `40000`
- Device: `cuda`
- Checkpoint: `checkpoints/run_2000_2008.pt`
- Evaluation table: `checkpoints/eval.csv`

## What Worked

- Training completed without divergence, `nan`, or `inf`.
- Physics losses converged to small values:
  - `L_rsf ~ 4.5e-06`
  - `L_state ~ 1.1e-05`
- Background stress stayed in a plausible range:
  - `tau_0_mean ~ 5.99e+07 Pa`
  - `tau_dot_mean ~ 6.6e-04 Pa/s`
- Predict snapshots from `2000` to `2008` evolved smoothly.
- `tau_elastic` and `tau_rsf` stayed close in prediction snapshots.
- The model produced stable fault variables and no obvious numerical instability.

## What Did Not Work Well

- GNSS fit is only moderate, not tight:
  - `global_rmse_mm = 13.42`
  - `global_mae_mm = 7.76`
- Error grows with time:
  - `2000` yearly mean RMSE: about `5.55 mm`
  - `2008` yearly mean RMSE: about `14.61 mm`
- Error increases strongly when more stations are available.
- `D_c` is concentrated near the upper allowed bound:
  - about `0.034 - 0.047 m`
- `(a-b)` remains close to zero and does not yet show strong spatial structure.

## Interpretation

- As a first inverse-physics run, the model is good: it is stable and physically coherent.
- As a full GNSS fit, the model is not yet good enough.
- The model captures the large-scale interseismic behaviour better than local details.
- The later years are more difficult, partly because the GNSS network becomes denser and partly because the state approaches the 2009 earthquake.

## Key Metrics

- Evaluation time range: `2000-01-01` to `2008-12-31`
- Time samples evaluated: `3288`
- Total valid components: `253896`
- Median time RMSE: `8.81 mm`
- P90 time RMSE: `14.91 mm`
- Worst time RMSE: `18.48 mm`
- Mean valid stations per timestep: `25.74`
- Median valid stations per timestep: `9`

## Next Steps

1. Inspect residuals by station and by component (`E`, `N`, `U`).
2. Add or test a velocity-based data term (`K_cd @ V`) to connect with the MLP.
3. Understand whether `D_c` is saturating because of bounds or weak identifiability.
4. Compare patch-wise fields (`s`, `V`, `tau_elastic`, `tau_rsf`) at `2000`, `2004`, and `2008`.
