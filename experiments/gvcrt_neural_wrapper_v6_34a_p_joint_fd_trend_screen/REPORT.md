# v6.34a trend screen report

Execution status: **PASS**.

- Mode: A (reuse); added training updates: N=0, P=0; fixed endpoint: step 250.
- Logical RD points: 90/90 (90 fresh physical encodes; 36 step-0 branch aliases point to 18 shared original encodes).
- Six fixed U-Long training-pool holdout videos; QP 0/5/9; 64 frames each.
- Real RANS streams, independent decode, and state synchronization are checked for each physical point.
- Rate trend uses actual measured bpp and no-extrapolation interpolation. FID uses one pooled 384-frame feature set per method/QP.

## Same-rate metric summaries

- N_noP_BG_FD_100 vs original | LPIPS: status=available, delta=-0.00019568016050970348, relative_change_percent=-0.12447156807458307, videos=6/6.
- N_noP_BG_FD_100 vs original | DISTS: status=available, delta=0.0004855523881746497, relative_change_percent=0.45352992386165586, videos=6/6.
- N_noP_BG_FD_100 vs original | FloLPIPS: status=available, delta=-0.000942936699560126, relative_change_percent=-0.5793502310896186, videos=6/6.
- N_noP_BG_FD_250 vs original | LPIPS: status=available, delta=-0.002248732435690306, relative_change_percent=-1.4304120137753384, videos=6/6.
- N_noP_BG_FD_250 vs original | DISTS: status=available, delta=-0.003321347833785525, relative_change_percent=-3.102302999348082, videos=6/6.
- N_noP_BG_FD_250 vs original | FloLPIPS: status=available, delta=-0.0019572290009853806, relative_change_percent=-1.2025420948671723, videos=6/6.
- P_PBG_FD_100 vs original | LPIPS: status=available, delta=-0.0003626154228274483, relative_change_percent=-0.23075795019325068, videos=6/6.
- P_PBG_FD_100 vs original | DISTS: status=available, delta=0.0006835601008532984, relative_change_percent=0.6386579434677787, videos=6/6.
- P_PBG_FD_100 vs original | FloLPIPS: status=available, delta=-0.0013794758291157446, relative_change_percent=-0.8480159089503824, videos=6/6.
- P_PBG_FD_250 vs original | LPIPS: status=available, delta=-0.0027451925535805183, relative_change_percent=-1.7468646119989566, videos=6/6.
- P_PBG_FD_250 vs original | DISTS: status=available, delta=-0.002465366529091311, relative_change_percent=-2.3032878151476806, videos=6/6.
- P_PBG_FD_250 vs original | FloLPIPS: status=available, delta=-0.002085400326952819, relative_change_percent=-1.281858866376412, videos=6/6.
- P_PBG_FD_100 vs N_noP_BG_FD_100 | LPIPS: status=available, delta=-0.000169667117815717, relative_change_percent=-0.10810397587359223, videos=6/6.
- P_PBG_FD_100 vs N_noP_BG_FD_100 | DISTS: status=available, delta=0.0001960525951402141, relative_change_percent=0.1823436143193252, videos=6/6.
- P_PBG_FD_100 vs N_noP_BG_FD_100 | FloLPIPS: status=available, delta=-0.0004393546347799338, relative_change_percent=-0.27165787987546414, videos=6/6.
- P_PBG_FD_250 vs N_noP_BG_FD_250 | LPIPS: status=available, delta=-0.0004955054807904154, relative_change_percent=-0.31988730090805584, videos=6/6.
- P_PBG_FD_250 vs N_noP_BG_FD_250 | DISTS: status=available, delta=0.000857355133626489, relative_change_percent=0.8266522770243782, videos=6/6.
- P_PBG_FD_250 vs N_noP_BG_FD_250 | FloLPIPS: status=available, delta=-0.00012649550847380347, relative_change_percent=-0.07870221884163202, videos=6/6.
- N_noP_BG_FD_100 vs original | FID: status=available, delta=-0.8581199529882462, relative_change_percent=-2.576781985274501, videos=1/1.
- N_noP_BG_FD_250 vs original | FID: status=available, delta=-2.0918118376937644, relative_change_percent=-6.281339853691796, videos=1/1.
- P_PBG_FD_100 vs original | FID: status=available, delta=-1.1440089793419095, relative_change_percent=-3.4352560136786194, videos=1/1.
- P_PBG_FD_250 vs original | FID: status=available, delta=-2.673633387445652, relative_change_percent=-8.028446750372607, videos=1/1.
- P_PBG_FD_100 vs N_noP_BG_FD_100 | FID: status=available, delta=-0.2858890263536635, relative_change_percent=-0.8811801189674932, videos=1/1.
- P_PBG_FD_250 vs N_noP_BG_FD_250 | FID: status=available, delta=-0.5818215497518878, relative_change_percent=-1.8642038778118772, videos=1/1.

## Pooled FID points

- original QP0: FID=60.540334, D_mu=9.6104879, samples=384, mean_actual_bpp=0.0020528107.
- original QP5: FID=23.016519, D_mu=3.0031614, samples=384, mean_actual_bpp=0.0077126535.
- original QP9: FID=14.452219, D_mu=1.5920308, samples=384, mean_actual_bpp=0.017225588.
- N_noP_BG_FD_100 QP0: FID=58.599391, D_mu=9.2481661, samples=384, mean_actual_bpp=0.0020528107.
- N_noP_BG_FD_100 QP5: FID=22.736784, D_mu=2.8925896, samples=384, mean_actual_bpp=0.0077126535.
- N_noP_BG_FD_100 QP9: FID=13.951258, D_mu=1.4674089, samples=384, mean_actual_bpp=0.017225588.
- N_noP_BG_FD_250 QP0: FID=56.576253, D_mu=8.7367344, samples=384, mean_actual_bpp=0.0020528107.
- N_noP_BG_FD_250 QP5: FID=21.81627, D_mu=2.661972, samples=384, mean_actual_bpp=0.0077126535.
- N_noP_BG_FD_250 QP9: FID=13.229009, D_mu=1.3202167, samples=384, mean_actual_bpp=0.017225588.
- P_PBG_FD_100 QP0: FID=58.386578, D_mu=9.251173, samples=384, mean_actual_bpp=0.0020508717.
- P_PBG_FD_100 QP5: FID=22.140143, D_mu=2.7486572, samples=384, mean_actual_bpp=0.0077080721.
- P_PBG_FD_100 QP9: FID=14.325009, D_mu=1.5425588, samples=384, mean_actual_bpp=0.017248887.
- P_PBG_FD_250 QP0: FID=54.935532, D_mu=8.4302979, samples=384, mean_actual_bpp=0.0020507109.
- P_PBG_FD_250 QP5: FID=21.708713, D_mu=2.6953757, samples=384, mean_actual_bpp=0.007710624.
- P_PBG_FD_250 QP9: FID=13.227971, D_mu=1.3141673, samples=384, mean_actual_bpp=0.017277541.

This is a six-video, three-QP, single-seed short trend screen. It does not claim convergence or broad generalization. No result-based checkpoint selection or budget extension was performed.

## Publication status

Local commit `6a0a25cfd078e3252f85c90bcebebc078bb1b878` contains the completed results. Push to the configured `origin` was not confirmed: `git push` timed out after 30 seconds and `git ls-remote` timed out after 20 seconds; the configured SSH alias resolves to `github.com`, for which host DNS lookup returned no address. No force push was attempted. See `publication_status.json`.
