Related Codes
=============

SPECTRAX-GK is designed to interoperate conceptually with established
gyrokinetic codes and to reproduce their benchmark cases. Key reference codes
include:

- **GX**: GPU-native gyrokinetic turbulence code with Laguerre-Hermite velocity
  space and field-aligned geometry. [GX]_
- **GENE**: Eulerian gyrokinetic code used for extensive verification and
  validation studies. [GENE]_
- **stella**: Radially local gyrokinetic solver for stellarators and tokamaks.
  [STELLA]_

We will use these codes (and their published benchmarks) as validation targets
for linear and nonlinear regimes.

Current linear benchmarking policy:

- Cyclone, ETG: compare against GS2 and stella.
- KBM: use GX (s-alpha geometry) as the primary electromagnetic cross-code baseline.
- stella KBM comparisons are reported as diagnostic-only until the documented
  ``beta/fapar/fbpar`` behavior is unambiguously electromagnetic in the tested
  stella build. [STELLADOCS]_
