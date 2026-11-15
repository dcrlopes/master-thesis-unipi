# wks720 smoke results, 15 Nov 2026

Copied from wks720 over SSH, which has no GitHub access since 15 Nov 2026. These runs locate the OpenMC
heap-corruption aborts of 3 Nov 2026. The OpenMC run directories (statepoints, 4.7 GB and 5.7 GB) remain on
wks720 only. The Mac mini Campaign 11 smoke of the same night is at the repository root
(`out_c11_smoke_mac.log`, `mac_smoke.out`, `out_c11_smoke_mac/`).

| Folder | Run | Code | OpenMC | Threads | Sessions | Aborts | Wall time |
|---|---|---|---|---|---|---|---|
| `c11_smoke_ab/` | Campaign 11 smoke (`ab_smoke.sh`), 00:48 to 02:20 | a6f8668 | 0.15.3 | 64 | 30 | 0 | 91 min |
| `c12_smoke/` | Campaign 12 smoke (`run_c12.sh --smoke`), 02:55 to 04:16 | 08d7c0f | 0.15.3 | 64 | 29 | 0 | 82 min |
| `c12_smoke_2026-11-03/` | Campaign 12 smoke attempts of 3 Nov (`c12.log`: 13:20 to 13:58; `out_c12_smoke_2026-11-03.log`: 15:42 to 16:00) | 08d7c0f and earlier | 0.15.3 | 64 | 13 and 9 | 1 and 2 | aborted |
| `c12/` | Campaign 12 (`run_c12.sh`), 05:17 to 21:26: 99 seed designs + 3 batches of 6, stopped by the hypervolume rule at 117 evaluations | e464486 | 0.15.3 | 64 | 90 | 0 | 16 h 09 min |
| repository root, `*_mac*` | Campaign 11 smoke on the Mac mini, 03:57 to 04:33 | 4fe70e0 | 0.16.0 | 12 | 34 | 0 | 36 min |

Sessions are the OpenMC banners in each log. The Campaign 12 smoke stops every 3D depletion at the mission
burnup (18.23 MWd/kgHM), so its `efpd_3d` is 1826 d by construction and is not comparable with the Campaign 11
cycle lengths.

`c12/` holds the Campaign 12 checkpoint (`out_c12/`), the driver and campaign logs, the machine log, the seed and
kappa files written by the campaign, and the evaluation cache (`openmc_runs_c12/eval_cache.json` on wks720). The
`c12_kappa.json` at the repository root is the pre-campaign file and is unchanged; the copy in `c12/` is the one
the campaign used.
