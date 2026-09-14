import { fromArrayBuffer } from "geotiff";
import proj4 from "proj4";

const HF = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main";

export interface SampleProps {
  dataset: string;
  sensor: string;
  country: string;
  date: string;
  id: string;
  file: string;
  flux?: number | null;
}

export type Asset = "target" | "ch4" | "plume";

interface Loaded {
  key: string;
  title: string;
  subtitle: string;
  dataUrl: string;
  coordinates: [number, number][];
  visible: boolean;
}

interface UiElements {
  root: HTMLElement;
  body: HTMLElement;
}

const loaded: Loaded[] = [];

function utm(zone: number, south: boolean): string {
  return `+proj=utm +zone=${zone}${south ? " +south" : ""} +datum=WGS84 +units=m +no_defs`;
}

function projFromGeoKeys(keys: Record<string, number | string>): string | null {
  const code = Number(keys.ProjectedCSTypeGeoKey || 0);
  if (code === 4326 || code === 4979) return "EPSG:4326";
  if (code >= 32601 && code <= 32660) return utm(code - 32600, false);
  if (code >= 32701 && code <= 32760) return utm(code - 32700, true);

  const citation = String(keys.GTCitationGeoKey || "");
  const match = citation.match(/UTM zone\s+(\d+)\s*([NS])/i);
  if (match) return utm(Number(match[1]), match[2].toUpperCase() === "S");

  const projection = Number(keys.ProjectionGeoKey || 0);
  if (projection >= 16001 && projection <= 16060) return utm(projection - 16000, false);
  if (projection >= 16101 && projection <= 16160) return utm(projection - 16100, true);

  const geo = Number(keys.GeographicTypeGeoKey || 0);
  if (geo === 4326) return "EPSG:4326";
  return null;
}

function colormap(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0, [13, 8, 135]],
    [0.25, [126, 3, 168]],
    [0.5, [204, 71, 120]],
    [0.75, [248, 149, 64]],
    [1, [252, 255, 164]],
  ];
  const v = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (v >= p0 && v <= p1) {
      const f = (v - p0) / (p1 - p0 || 1);
      return [0, 1, 2].map((k) => Math.round(c0[k] + (c1[k] - c0[k]) * f)) as [number, number, number];
    }
  }
  return stops[stops.length - 1][1];
}

async function fetchRange(url: string, start: number, end: number): Promise<Uint8Array> {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (res.status !== 206 && res.status !== 200) throw new Error(`Range failed: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function extractTacozipEntry(url: string, entryName: string): Promise<Uint8Array> {
  const tailRes = await fetch(url, { headers: { Range: "bytes=-131072" } });
  const tail = new Uint8Array(await tailRes.arrayBuffer());
  const range = tailRes.headers.get("content-range") || "";
  const total = Number(range.split("/")[1] || 0);
  const tailStart = total - tail.length;

  const view = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("ZIP end-of-central-directory not found");

  const cdSize = view.getUint32(eocd + 12, true);
  const cdOffset = view.getUint32(eocd + 16, true);

  const cd = cdOffset >= tailStart ? tail.subarray(cdOffset - tailStart) : await fetchRange(url, cdOffset, cdOffset + cdSize - 1);
  const cdv = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);

  let p = 0;
  let found: { method: number; compSize: number; localOffset: number } | null = null;
  while (p + 46 <= cd.length) {
    if (cdv.getUint32(p, true) !== 0x02014b50) break;
    const method = cdv.getUint16(p + 10, true);
    const compSize = cdv.getUint32(p + 20, true);
    const nameLen = cdv.getUint16(p + 28, true);
    const extraLen = cdv.getUint16(p + 30, true);
    const commentLen = cdv.getUint16(p + 32, true);
    const localOffset = cdv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(cd.subarray(p + 46, p + 46 + nameLen));
    if (name === entryName) {
      found = { method, compSize, localOffset };
      break;
    }
    p += 46 + nameLen + extraLen + commentLen;
  }
  if (!found) throw new Error(`Entry not found: ${entryName}`);

  const localHeader = await fetchRange(url, found.localOffset, found.localOffset + 29);
  const lv = new DataView(localHeader.buffer, localHeader.byteOffset, localHeader.byteLength);
  const dataStart = found.localOffset + 30 + lv.getUint16(26, true) + lv.getUint16(28, true);
  const raw = await fetchRange(url, dataStart, dataStart + found.compSize - 1);
  return found.method === 0 ? raw : await inflateRaw(raw);
}

function percentiles(data: ArrayLike<number>): { lo: number; hi: number } {
  const arr = Array.from(data).sort((a, b) => a - b);
  const at = (q: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.floor(arr.length * q)))];
  return { lo: at(0.02), hi: at(0.98) };
}

async function renderAsset(bytes: Uint8Array, asset: Asset) {
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const image = await (await fromArrayBuffer(buffer)).getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const count = image.getSamplesPerPixel();

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);

  if (asset === "target") {
    const samples = count >= 13 ? [11, 7, 3] : count >= 11 ? [6, 4, 3] : [0];
    const bands = (await image.readRasters({ samples })) as unknown as ArrayLike<number>[];
    const scales = bands.map(percentiles);
    for (let i = 0; i < width * height; i++) {
      for (let c = 0; c < 3; c++) {
        const band = bands[Math.min(c, bands.length - 1)];
        const s = scales[Math.min(c, scales.length - 1)];
        const v = s.hi === s.lo ? 0 : (band[i] - s.lo) / (s.hi - s.lo);
        img.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
      }
      img.data[i * 4 + 3] = 255;
    }
  } else {
    const band = ((await image.readRasters()) as unknown as ArrayLike<number>[])[0];
    const { hi } = percentiles(band);
    for (let i = 0; i < width * height; i++) {
      const t = hi === 0 ? 0 : band[i] / hi;
      if (asset === "ch4") {
        const [r, g, b] = colormap(t);
        img.data[i * 4] = r;
        img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = Math.round(Math.min(1, Math.max(0, (t - 0.1) / 0.3)) * 255);
      } else {
        const on = band[i] > 0;
        img.data[i * 4] = 249;
        img.data[i * 4 + 1] = 115;
        img.data[i * 4 + 2] = 22;
        img.data[i * 4 + 3] = on ? 210 : 0;
      }
    }
  }
  ctx.putImageData(img, 0, 0);

  const geoKeys = image.getGeoKeys() as Record<string, number | string>;
  const proj = projFromGeoKeys(geoKeys);
  const [minX, minY, maxX, maxY] = image.getBoundingBox();
  const corners: [number, number][] = [
    [minX, maxY],
    [maxX, maxY],
    [maxX, minY],
    [minX, minY],
  ];
  const coordinates = corners.map(([x, y]) => {
    const [lng, lat] = proj ? proj4(proj, "EPSG:4326", [x, y]) : [x, y];
    if (!isFinite(lat) || Math.abs(lat) > 90 || Math.abs(lng) > 180) throw new Error("cannot georeference this sample");
    return [lng, lat] as [number, number];
  });

  return { dataUrl: canvas.toDataURL("image/png"), coordinates, width, height };
}

const ASSET_LABEL: Record<Asset, string> = { target: "RGB", ch4: "CH₄ enhancement", plume: "Plume mask" };

function layerId(key: string, suffix: string): string {
  return `${suffix}-${key}`;
}

function addToMap(map: any, item: Loaded): void {
  const src = layerId(item.key, "samplesrc");
  const raster = layerId(item.key, "sampleimg");
  const outlineSrc = layerId(item.key, "outlinesrc");
  const outline = layerId(item.key, "outline");
  if (map.getSource(src)) return;

  map.addSource(src, { type: "image", url: item.dataUrl, coordinates: item.coordinates });
  map.addLayer({
    id: raster,
    type: "raster",
    source: src,
    paint: { "raster-opacity": 0.95 },
    layout: { visibility: item.visible ? "visible" : "none" },
  });
  map.addSource(outlineSrc, {
    type: "geojson",
    data: { type: "Feature", geometry: { type: "Polygon", coordinates: [[...item.coordinates, item.coordinates[0]]] }, properties: {} },
  });
  map.addLayer({
    id: outline,
    type: "line",
    source: outlineSrc,
    paint: { "line-color": "#fde047", "line-width": 1.2, "line-opacity": 0.85 },
    layout: { visibility: item.visible ? "visible" : "none" },
  });
}

function removeFromMap(map: any, key: string): void {
  for (const suffix of ["sampleimg", "outline"]) {
    if (map.getLayer(layerId(key, suffix))) map.removeLayer(layerId(key, suffix));
  }
  for (const suffix of ["samplesrc", "outlinesrc"]) {
    if (map.getSource(layerId(key, suffix))) map.removeSource(layerId(key, suffix));
  }
}

function renderList(map: any, ui: UiElements): void {
  if (loaded.length === 0) {
    ui.root.style.display = "none";
    return;
  }
  ui.root.style.display = "block";
  ui.body.innerHTML =
    `<p class="globe-panel__title">Layers · ${loaded.length}</p>` +
    loaded
      .map(
        (item) =>
          `<div class="sv-item">` +
          `<input type="checkbox" data-toggle="${item.key}" ${item.visible ? "checked" : ""}>` +
          `<div class="sv-item__text"><b>${item.title}</b><span>${item.subtitle}</span></div>` +
          `<button class="sv-remove" data-remove="${item.key}" type="button" aria-label="Remove">×</button>` +
          `</div>`,
      )
      .join("");

  ui.body.querySelectorAll<HTMLInputElement>("[data-toggle]").forEach((el) => {
    el.addEventListener("change", () => {
      const item = loaded.find((x) => x.key === el.dataset.toggle);
      if (!item) return;
      item.visible = el.checked;
      const vis = el.checked ? "visible" : "none";
      for (const suffix of ["sampleimg", "outline"]) {
        if (map.getLayer(layerId(item.key, suffix))) map.setLayoutProperty(layerId(item.key, suffix), "visibility", vis);
      }
    });
  });
  ui.body.querySelectorAll<HTMLButtonElement>("[data-remove]").forEach((el) => {
    el.addEventListener("click", () => {
      const key = el.dataset.remove!;
      removeFromMap(map, key);
      const i = loaded.findIndex((x) => x.key === key);
      if (i >= 0) loaded.splice(i, 1);
      renderList(map, ui);
    });
  });
}

export function reAddAll(map: any): void {
  for (const item of loaded) {
    try {
      addToMap(map, item);
    } catch (e) {
      /* style not ready */
    }
  }
}

export function showChooser(map: any, props: SampleProps, ui: UiElements): void {
  ui.root.style.display = "block";
  const assets: Asset[] = ["target", "ch4", "plume"];
  ui.body.innerHTML =
    `<p class="globe-panel__title">Load sample</p>` +
    `<p class="sv-title">${props.sensor} · ${props.country}</p>` +
    `<p class="sv-status">${props.date} · ${props.dataset}</p>` +
    `<div class="sv-assets">` +
    assets
      .map((a) => `<button class="sv-asset" data-asset="${a}" type="button">${ASSET_LABEL[a]}</button>`)
      .join("") +
    `</div>` +
    (loaded.length ? `<button class="sv-clear" data-showlist type="button">Show layers (${loaded.length})</button>` : "");

  ui.body.querySelectorAll<HTMLButtonElement>("[data-asset]").forEach((el) => {
    el.addEventListener("click", () => loadAsset(map, props, el.dataset.asset as Asset, ui));
  });
  ui.body.querySelector<HTMLButtonElement>("[data-showlist]")?.addEventListener("click", () => renderList(map, ui));
}

export async function loadAsset(map: any, props: SampleProps, asset: Asset, ui: UiElements): Promise<void> {
  ui.root.style.display = "block";
  ui.body.innerHTML = `<p class="globe-panel__title">Loading</p><p class="sv-status">Reading ${ASSET_LABEL[asset]} from Hugging Face…</p>`;

  const url = `${HF}/${props.dataset}/${props.file}`;
  try {
    const bytes = await extractTacozipEntry(url, `DATA/${props.id}/${asset}`);
    const { dataUrl, coordinates, width, height } = await renderAsset(bytes, asset);
    const key = `${props.id.slice(0, 8)}-${asset}-${Date.now().toString(36)}`;
    const item: Loaded = {
      key,
      title: `${ASSET_LABEL[asset]} · ${props.sensor}`,
      subtitle: `${props.country} · ${props.date} · ${width}×${height}`,
      dataUrl,
      coordinates,
      visible: true,
    };
    loaded.push(item);
    addToMap(map, item);

    const lngs = coordinates.map((c) => c[0]);
    const lats = coordinates.map((c) => c[1]);
    map.fitBounds(
      [
        [Math.min(...lngs), Math.min(...lats)],
        [Math.max(...lngs), Math.max(...lats)],
      ],
      { padding: { top: 120, bottom: 120, left: 420, right: 360 }, duration: 900 },
    );

    renderList(map, ui);
    (window as any).__sampleLoaded = true;
  } catch (err) {
    ui.body.innerHTML =
      `<p class="globe-panel__title">Error</p>` +
      `<p class="sv-status sv-error">Could not load ${ASSET_LABEL[asset]}. ${(err as Error).message}</p>`;
    (window as any).__sampleError = (err as Error).message;
    console.error(err);
  }
}
