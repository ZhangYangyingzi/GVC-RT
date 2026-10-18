# V6.34b completion report

Execution status: PASS.

- N cumulative updates: 500 (continued from 475).
- P cumulative updates: 500 (continued from 368).
- Reused historical points: 90, with source receipts and artifact hashes checked.
- New N500/P500 real encodes: 36.
- Total logical points including aliases: 126.
- Rate comparisons use the seven-model shared measured-bpp interval where available, 20 linear points in ln(bpp), and no extrapolation. Pooled FID uses the six-video, 384-frame pool.
- Input is a fixed six-video U-Long training-pool holdout screen at QP 0/5/9; this is a single-seed short trend screen. No checkpoint selection or budget extension was performed.

## Equal-rate comparison rows

- Video 0 LPIPS: delta=0.0011779846; relative_change_percent=1.4960151037467677; common_bpp=[0.0027252725,0.021045223].
- Video 0 DISTS: delta=0.00099372862; relative_change_percent=1.6383520133224152; common_bpp=[0.0027252725,0.021045223].
- Video 0 FloLPIPS: delta=0.0013740819; relative_change_percent=1.5522835893164233; common_bpp=[0.0027252725,0.021045223].
- Video 1 LPIPS: delta=-0.0041363256; relative_change_percent=-1.8114249569554224; common_bpp=[0.0014131221,0.011868248].
- Video 1 DISTS: delta=-0.00086842566; relative_change_percent=-0.7456886900749775; common_bpp=[0.0014131221,0.011868248].
- Video 1 FloLPIPS: delta=-0.0043954515; relative_change_percent=-1.7887665268432587; common_bpp=[0.0014131221,0.011868248].
- Video 2 LPIPS: delta=0.0013851793; relative_change_percent=0.8270326475521143; common_bpp=[0.0015025801,0.0088198664].
- Video 2 DISTS: delta=0.00022504815; relative_change_percent=0.24779393538510788; common_bpp=[0.0015025801,0.0088198664].
- Video 2 FloLPIPS: delta=0.001665129; relative_change_percent=0.9668595552905724; common_bpp=[0.0015025801,0.0088198664].
- Video 3 LPIPS: delta=0.00042060381; relative_change_percent=0.5352456015323556; common_bpp=[0.0024459877,0.019703897].
- Video 3 DISTS: delta=4.0542057e-05; relative_change_percent=0.047943523218586664; common_bpp=[0.0024459877,0.019703897].
- Video 3 FloLPIPS: delta=-0.0001018181; relative_change_percent=-0.17844227217407624; common_bpp=[0.0024459877,0.019703897].
- Video 4 LPIPS: delta=0.0021458738; relative_change_percent=1.753631605933726; common_bpp=[0.0010599923,0.0094332923].
- Video 4 DISTS: delta=0.0040801263; relative_change_percent=3.5938151110184124; common_bpp=[0.0010599923,0.0094332923].
- Video 4 FloLPIPS: delta=0.0037313149; relative_change_percent=3.121212017681037; common_bpp=[0.0010599923,0.0094332923].
- Video 5 LPIPS: delta=0.00072534785; relative_change_percent=0.3464635204349784; common_bpp=[0.0032021002,0.032397823].
- Video 5 DISTS: delta=0.00049501485; relative_change_percent=0.3707934530076728; common_bpp=[0.0032021002,0.032397823].
- Video 5 FloLPIPS: delta=0.00075417885; relative_change_percent=0.3225234891602625; common_bpp=[0.0032021002,0.032397823].

## Pooled FID comparison

{'method': 'P_PBG_FD_500', 'reference': 'N_noP_BG_FD_500', 'metric': 'FID', 'status': 'available', 'method_mean': 30.451398425692283, 'reference_mean': 30.07323249686783, 'delta': 0.3781659288244558, 'relative_change_percent': 1.257483474261845, 'better_grid_fraction': 0.2, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'N_noP_BG_FD_500', 'reference': 'original', 'metric': 'FID', 'status': 'available', 'method_mean': 30.07323249686783, 'reference_mean': 33.3020006307143, 'delta': -3.2287681338464878, 'relative_change_percent': -9.695417910924569, 'better_grid_fraction': 1.0, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'P_PBG_FD_500', 'reference': 'original', 'metric': 'FID', 'status': 'available', 'method_mean': 30.451398425692283, 'reference_mean': 33.3020006307143, 'delta': -2.850602205022032, 'relative_change_percent': -8.559852714653225, 'better_grid_fraction': 1.0, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'N_noP_BG_FD_100', 'reference': 'P_PBG_FD_100', 'metric': 'FID', 'status': 'available', 'method_mean': 32.44388067772606, 'reference_mean': 32.1579916513724, 'delta': 0.2858890263536635, 'relative_change_percent': 0.8890139329999567, 'better_grid_fraction': 0.15, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'N_noP_BG_FD_250', 'reference': 'P_PBG_FD_250', 'metric': 'FID', 'status': 'available', 'method_mean': 31.210188793020553, 'reference_mean': 30.628367243268656, 'delta': 0.5818215497518878, 'relative_change_percent': 1.8996166042111093, 'better_grid_fraction': 0.1, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'N_noP_BG_FD_500', 'reference': 'P_PBG_FD_500', 'metric': 'FID', 'status': 'available', 'method_mean': 30.07323249686783, 'reference_mean': 30.451398425692283, 'delta': -0.3781659288244558, 'relative_change_percent': -1.2418671994564026, 'better_grid_fraction': 0.8, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'N_noP_BG_FD_500', 'reference': 'N_noP_BG_FD_250', 'metric': 'FID', 'status': 'available', 'method_mean': 30.07323249686783, 'reference_mean': 31.210188793020553, 'delta': -1.1369562961527233, 'relative_change_percent': -3.642901052899038, 'better_grid_fraction': 1.0, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}
{'method': 'P_PBG_FD_500', 'reference': 'P_PBG_FD_250', 'metric': 'FID', 'status': 'available', 'method_mean': 30.451398425692283, 'reference_mean': 30.628367243268656, 'delta': -0.17696881757637967, 'relative_change_percent': -0.5777938346200057, 'better_grid_fraction': 0.7, 'common_mean_bpp_min': 0.002052810731738683, 'common_mean_bpp_max': 0.01722558794688786, 'aggregation': 'pooled six-video Inception FID; linear trend'}

All source points, reused receipts, new point receipts, and evaluation artifacts are indexed in this directory.

## Continuation transaction checks

- Parent complete transactions restored: N=475 and P=368.
- New committed AdamW updates: N=25 and P=132; both cumulative endpoints are 500.
- New continuation optimizer call ledger: STARTED/APPLIED/COMMITTED counts match each required added update count.
- Both EMA ledgers end at 500 commits and match the first 500 shared plan stratum counts.
- New N/P endpoint points have valid task signatures, complete artifacts, real RANS, independent decode, and reference synchronization.
- Resolved startup file lookup failures are retained in `errors.jsonl` and documented in `audits/startup_repair.json`.
