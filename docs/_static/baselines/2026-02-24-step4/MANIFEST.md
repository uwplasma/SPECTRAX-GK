## Baseline Manifest (Step 4)

- Timestamp (UTC): `2026-02-24T05:42:57Z`
- Git commit (short): `fca003d`
- Git commit (full): `fca003d29e7f98b6674f2686f769b3510acf1589`

### Scope

Step 4 delivered:

- normalization/sign contract documentation updates
- full post-change artifact regeneration (tables + figures)
- full validation gates (tests, type-check, docs build)

### Commands run

```bash
python tools/make_tables.py --case all --no-progress
python tools/make_figures.py --case all --no-progress
pytest -q --maxfail=1 --disable-warnings
mypy src/spectraxgk
python -m sphinx -W -b html docs docs/_build/html
```

### Artifact checksums (SHA-256)

```text
051f2d108d95fe679425110cd49032e128bc22c7d998ea80d55b9c17805e97ad  kbm_mismatch_table.csv
0ab56cbcb070bb15ad988ff6c074c55bbe235d122e87cadf509ccfa17413c040  etg_mismatch_table.csv
1d751fdf1a507d6611c505354d5ff6d6c8fd57434c8db7c1c5ddfb2aaff4d047  kinetic_mismatch_table.csv
2baf3b30b4c7ae5147a93f1574ebfb7235830ce1e0acb4f0793df11190b4329c  linear_summary.png
3904d7dc98023f1f7cba82ee6cd25e90ba6d066ae7f543ce4e18ea8e25ce5bf4  cyclone_reference.pdf
468106d761d6c7bcf0cdc0ee1b52dace75878c773c5f9fc834a2474150ae6319  linear_summary.pdf
4b400175d0f3ee35e00f6636fcb74bd315c0c46ac884893b35c543c22407f5c7  tem_mismatch_table.csv
53b3c3371e70a1f83cbea72f9350cc15df2648fd133258b6dc30a008ba396495  cyclone_mismatch_table.csv
665186718c420bfb0da1597070f98bfc11c5481a4b2219f4e73ee450a7db807a  etg_comparison.png
6f27772e1250aeb5185d9b5db6b51983f5519fc38900ebf720c25ade9c68d6c5  etg_comparison.pdf
c70a6e0202899af37c32a6acc9b440af9f940b9e3811c581f232cbf599bf7ba0  cyclone_comparison.pdf
d337ecbafddfa852536d0f701374d90a0cd981a6d491338c27761aeb074caf79  cyclone_comparison.png
eecebf9a730cfee2c41502717967431efd28b53631aadffa3cd080a79464af8f  etg_trend_table.csv
```
