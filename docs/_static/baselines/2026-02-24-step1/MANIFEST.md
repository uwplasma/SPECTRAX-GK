## Baseline Manifest (Step 1)

- Timestamp (UTC): `2026-02-24T04:06:23Z`
- Git commit (short): `111edf8`
- Git commit (full): `111edf8c6ee03256da2e116a3cc12a072c7bd4a3`

### Commands run

```bash
pytest -q --maxfail=1 --disable-warnings
mypy src/spectraxgk
python -m sphinx -W -b html docs docs/_build/html
python tools/make_tables.py --case cyclone --no-progress
python tools/make_tables.py --case etg --no-progress
python tools/make_figures.py --case cyclone --no-progress
python tools/make_figures.py --case etg --no-progress
```

### Artifact checksums (SHA-256)

```text
0a0a8dff3691e0943f2b6e441066e6d655dcd26c5e4eba439e155cf8159e3503  etg_stella_mismatch.csv
10c5e58cf0eeaf59aeb3768b2ba6225b5d57828dcf60d716abbab5e476c9a65d  etg_gs2_mismatch.csv
2baf3b30b4c7ae5147a93f1574ebfb7235830ce1e0acb4f0793df11190b4329c  linear_summary.png
362143681942973c2dba31784a659b29efe2341f7333400fa961c2e6de20e153  cyclone_full_operator_scan_table.csv
3cf1b09e9d2afbf676e3264cd35276565b5ae563e9bc841da808a69cded40198  etg_trend_table.csv
43a393aa499d14d9000a0491333d55938aeefd7c3300daffc014286732a1b328  cyclone_scan_convergence.csv
4b6ffa0ad6b9949edf4d29eaa8c6f21d886f3b64617b6dc07c903699cc6c0701  cyclone_comparison.png
4e6a00af6dd4f43138ab3ebb2290d814c970b6d51f7fa4ccdfe629a0472c8478  cyclone_comparison.pdf
53b3c3371e70a1f83cbea72f9350cc15df2648fd133258b6dc30a008ba396495  cyclone_mismatch_table.csv
5f628f5cedd6b9d90cb681a95a6af8ac869e3de2e11d36ace69ccef9373c85bb  etg_comparison.pdf
665186718c420bfb0da1597070f98bfc11c5481a4b2219f4e73ee450a7db807a  etg_comparison.png
8fa7640e75b465efb38dcc41e5db8a4ea82a7178a6eb3b5219edd81adc8c0780  cyclone_scan_table_highres.csv
8fa7640e75b465efb38dcc41e5db8a4ea82a7178a6eb3b5219edd81adc8c0780  cyclone_scan_table_lowres.csv
aa2bded9408e2dc7a371fc8e305297329407f2ed94af0a0cc1475c6fd8d7fae5  kbm_stella_mismatch.csv
ae8a66d4174c66eda3b9664cd82f236b3de2e70ecc66e0515fc41863ac7ebfe0  etg_mismatch_table.csv
b4f37abdedaaa7db9a73e3a99460c828f91e8282d3dc2af0165c15bccf047636  kbm_gs2_stella_comparison.png
b6f464ee53a6224112076c2470a843a2f6568843e4f54270fee25d746072f461  kbm_gs2_mismatch.csv
c041b981af494a6688a668f0b53f94c77efdd76e49621430cad6a8f743a462a7  cyclone_reference.pdf
```
