# Unzoned moderator-coefficient scans, superseded

These runs solved a UNIFORM core, because mtc_scan.py called make_core_model
without design_map. The campaign evaluator passes
design_map=zn.evaluator_design_map(design) at openmc_evaluator.py:406.

Measured consequence on design 47 at 1000 ppm and 570 K:

  unzoned  k = 1.13065   rho = 11555 pcm
  zoned    k = 1.09995   rho =  9087 pcm
  campaign k = 1.09805   rho =  8930 pcm   (580 K, density 0.72)

So zoning is worth about 2470 pcm and the remaining 157 pcm is the density
difference between the IAPWS value at 570 K and the 0.72 hardcoded in
make_water. The zoned scan agrees with the campaign model.

RETAINED, NOT DELETED. The coefficient itself barely moved, -24.87 +/- 2.15
zoned against -26.39 +/- 1.42 unzoned, so these runs measure the sensitivity
of the coefficient to radial zoning, which is a result in its own right.

Do NOT quote the crossings in these directories. Use the zoned reruns.
