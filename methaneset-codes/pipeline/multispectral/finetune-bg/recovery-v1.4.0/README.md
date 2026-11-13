# Recovery of the samples missing from the first release (v1.4.0)

The first release of the two multispectral finetune collections left 255 plumes out. This folder
holds the scripts that audited them, recovered the ones that could be recovered and repackaged the
TACOs without touching what was already published.

The rule is in `regla.py` and is frozen: three references per sample, no variable number of leaves.
A reference is discarded when more than 50 % of its pixels are cloud, cloud shadow or missing, and
candidates are restricted to the ±120 days around the target, which is the operational rule of
MARS-S2L.

| script | what it does |
|---|---|
| `t3_select.py` | selects three references for each missing sample from a pool downloaded with `01_download.py` |
| `t4_diagnostico.py` | writes the reason why a sample could not be recovered (no clean candidate, fewer than three, methane) |
| `t7_repack.py` | rebuilds the `.tacozip` files: published entries are copied byte by byte, new samples appended |
| `t8_fill.py` | fills the metadata columns of the new samples from the MARS catalog and the reference table |
| `t9_cards.py` | patches `COLLECTION.json`, `README.md` and `index.html` of each collection |
| `t10_upload.py` | uploads one collection per commit, so an interrupted upload never leaves a mixed dataset |

The card text of each collection lives in `cards/`. The diagnostic of the 255 samples is a CSV with
one row per sample: collection, sensor, id, candidates, how many pass the cloud threshold, median
cloud, median methane and the reason. Of the 255, 70 recovered (17 Landsat, 53 Sentinel-2) and 185
did not reach three references; the reasons are all in that table.
