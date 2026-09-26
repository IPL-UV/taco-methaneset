"""Quick exploration of the bank table (Julio's code, with the final column names).
For a notebook or `python -i explore_table.py`.
"""
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

R = "/data/databases/METHANESET_TACOS/methaneset-bank-raw/"
t = pd.read_parquet(R + "tabla.parquet")
print(t.shape)                                   # (336870, 39)
print(t.iloc[100000])
print(t[t["methane:sim_type"] == "multi"].iloc[1])
print(t.columns.tolist())

# one row per plume (drops the 855 sun positions)
m = t.drop_duplicates("methane:plume_uid")
print(len(m))                                    # 394
print(m.groupby("methane:wind_speed").size())    # plumes per wind speed
print(m.groupby("methane:emitter").size())       # per emitter

# criteria: west, south and north below 2% of the peak
print(m[["edge:west_frac", "edge:south_frac", "edge:north_frac", "edge:east_frac"]].describe())

# the ones that arrive the most loaded at the east edge
print(m.sort_values("edge:east_frac", ascending=False)[
    ["methane:plume_uid", "methane:peak_ppb", "edge:east_ppb", "edge:east_frac"]].head(10))

# all sun positions of one plume
p = t[t["methane:plume_uid"] == "p5_w07.0_s093"]
print(p[["id", "sun:sza", "sun:raa", "array:width", "array:height"]])

# open the GeoTIFF of one row
r = t.iloc[0]
with rasterio.open(R + r["path"]) as src:
    a = src.read(1)
    b = src.bounds
plt.imshow(a, extent=[b.left, b.right, b.bottom, b.top], cmap="magma")
plt.colorbar(label="ppb at 3 t/h")
plt.plot(0, 0, "c+")                             # source at (0, 0) m
plt.title(r["id"])
plt.show()

# the final TACO, read with tacoreader (deep environment)
import tacoreader
tacoreader.use("pandas")
ds = tacoreader.load("/data/databases/METHANESET_TACOS/methaneset-bank/")
table = ds.data
print(table.iloc[0])
