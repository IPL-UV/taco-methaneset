export interface Dataset {
  name: string;
  family: "Sentinel-2" | "Landsat 8/9" | "EMIT" | "Plume bank";
  subset: "Pretraining" | "Finetune" | "Single";
  samples: number;
  sizeGB: number;
  note: string;
}

export const datasets: Dataset[] = [
  {
    name: "methaneset-s2-pretraining",
    family: "Sentinel-2",
    subset: "Pretraining",
    samples: 57291,
    sizeGB: 37.2,
    note: "Confirmed plume-free scenes with a temporally paired reference.",
  },
  {
    name: "methaneset-s2-finetune",
    family: "Sentinel-2",
    subset: "Finetune",
    samples: 3612,
    sizeGB: 13.6,
    note: "Expert-verified plume masks over Sentinel-2 imagery.",
  },
  {
    name: "methaneset-l89-pretraining",
    family: "Landsat 8/9",
    subset: "Pretraining",
    samples: 21926,
    sizeGB: 8.88,
    note: "Confirmed plume-free scenes with a temporally paired reference.",
  },
  {
    name: "methaneset-l89-finetune",
    family: "Landsat 8/9",
    subset: "Finetune",
    samples: 1548,
    sizeGB: 0.78,
    note: "Expert-verified plume masks over Landsat 8/9 imagery.",
  },
  {
    name: "methaneset-emit",
    family: "EMIT",
    subset: "Single",
    samples: 721,
    sizeGB: 652,
    note: "Calibrated radiance cubes, matched-filter products and two independent masks.",
  },
  {
    name: "methaneset-bank",
    family: "Plume bank",
    subset: "Single",
    samples: 238545,
    sizeGB: 5.1,
    note: "Precomputed WRF-LES column enhancements across geometry and wind.",
  },
  {
    name: "methaneset-bank-les",
    family: "Plume bank",
    subset: "Single",
    samples: 1647,
    sizeGB: 3.1,
    note: "The raw 3D WRF-LES simulation cubes the bank is projected from.",
  },
];

export const totals = {
  datasets: datasets.length,
  samples: datasets.reduce((a, d) => a + d.samples, 0),
  sizeGB: datasets.reduce((a, d) => a + d.sizeGB, 0),
};

const hfBase = "https://huggingface.co/datasets/tacofoundation/methaneset/tree/main";

export const hfUrl = (name: string) => `${hfBase}/${name}`;
