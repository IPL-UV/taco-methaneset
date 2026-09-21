export interface Dataset {
  name: string;
  family: "Sentinel-2" | "Landsat 8/9" | "EMIT" | "Plume bank";
  subset: "Pretraining" | "Finetune" | "Single";
  samples: number;
  sizeGB: number;
  icon: "hyper" | "s2" | "l89" | "bank";
  abstract: string;
}

export const datasets: Dataset[] = [
  {
    name: "methaneset-s2-pretraining",
    family: "Sentinel-2",
    subset: "Pretraining",
    samples: 57291,
    sizeGB: 36.91,
    icon: "s2",
    abstract:
      "Confirmed plume-free Sentinel-2 scenes with a temporally close, plume-free reference. Pairs to combine with the plume bank for synthetic augmentation.",
  },
  {
    name: "methaneset-s2-finetune",
    family: "Sentinel-2",
    subset: "Finetune",
    samples: 3603,
    sizeGB: 6.37,
    icon: "s2",
    abstract:
      "Sentinel-2 imagery with expert-verified plume masks from IMEO MARS, all 13 bands at 10 m and 200x200 px patches.",
  },
  {
    name: "methaneset-l89-pretraining",
    family: "Landsat 8/9",
    subset: "Pretraining",
    samples: 21926,
    sizeGB: 8.88,
    icon: "l89",
    abstract:
      "Plume-free Landsat 8/9 scenes with a matched reference, extending the surface and temporal diversity back to 2018.",
  },
  {
    name: "methaneset-l89-finetune",
    family: "Landsat 8/9",
    subset: "Finetune",
    samples: 1353,
    sizeGB: 1.2,
    icon: "l89",
    abstract:
      "Landsat 8/9 imagery with expert-verified plume masks, 9 OLI bands at 30 m resampled to 10 m.",
  },
  {
    name: "methaneset-emit",
    family: "EMIT",
    subset: "Single",
    samples: 721,
    sizeGB: 1185.17,
    icon: "hyper",
    abstract:
      "EMIT radiance hypercubes (285 bands, 60 m) with standard matched filter, mag1c, and two independent plume masks per granule from IMEO and Carbon Mapper.",
  },
  {
    name: "methaneset-bank",
    family: "Plume bank",
    subset: "Single",
    samples: 238545,
    sizeGB: 5.08,
    icon: "bank",
    abstract:
      "Precomputed WRF-LES column enhancements across solar and wind geometry, at a reference rate of 3000 kg/h. Injected into any plume-free scene.",
  },
  {
    name: "methaneset-bank-les",
    family: "Plume bank",
    subset: "Single",
    samples: 1647,
    sizeGB: 3.06,
    icon: "bank",
    abstract:
      "The raw 3D WRF-LES simulation cubes the plume bank is projected from, for reprocessing under new geometries.",
  },
];

export const totals = {
  datasets: datasets.length,
  samples: datasets.reduce((a, d) => a + d.samples, 0),
  sizeGB: datasets.reduce((a, d) => a + d.sizeGB, 0),
};

export const fmtSize = (gb: number) =>
  gb >= 1000 ? `${(gb / 1000).toFixed(2)} TB` : `${gb} GB`;

export const HF_PREFIX = "hf://datasets/tacofoundation/methaneset";
const HF_BASE = "https://huggingface.co/datasets/tacofoundation/methaneset";

export const hfUrl = (name: string) => `${HF_BASE}/tree/main/${name}`;
export const hfReadme = (name: string) => `${HF_BASE}/blob/main/${name}/README.md`;
export const loadCode = (name: string) => `${HF_PREFIX}/${name}`;
