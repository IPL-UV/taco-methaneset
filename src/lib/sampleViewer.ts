import { fromArrayBuffer } from "geotiff";
import maplibregl from "maplibre-gl";
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
  viz?: string;
}

type Asset = "target" | "ch4" | "plume";

interface Feature {
  properties: SampleProps;
  geometry: { coordinates: [number, number] };
}

interface UiElements {
  root: HTMLElement;
  body: HTMLElement;
  layersRoot: HTMLElement;
  layersBody: HTMLElement;
  list?: Feature[];
}

interface Rendered {
  dataUrl: string;
  coordinates: [number, number][];
  width: number;
  height: number;
}

interface Entry {
  key: string;
  asset: Asset;
  props: SampleProps;
  rendered: Rendered;
  visible: boolean;
}

const ASSETS: Asset[] = ["target", "ch4", "plume"];
const ASSET_LABEL: Record<Asset, string> = {
  target: "RGB",
  ch4: "CH₄",
  plume: "Mask",
};

const cache = new Map<string, Rendered>();
const entries: Entry[] = [];
const loading = new Set<string>();
const failed = new Map<string, string>();
const bound = new Set<string>();

let mapRef: any = null;
let uiRef: UiElements | null = null;
let popup: any = null;
let curList: Feature[] = [];
let curIndex = 0;

const keyOf = (id: string, asset: Asset) => `${id}:${asset}`;
const ids = (key: string) => {
  const s = key.replace(/[^a-zA-Z0-9]/g, "-");
  return { src: `ssrc-${s}`, img: `simg-${s}`, osrc: `osrc-${s}`, fill: `sfill-${s}` };
};

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

async function fetchWithTimeout(url: string, opts: RequestInit, ms = 30000): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchRange(url: string, start: number, end: number): Promise<Uint8Array> {
  const res = await fetchWithTimeout(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (res.status !== 206 && res.status !== 200) throw new Error(`Range failed: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function withRetry<T>(fn: () => Promise<T>, tries = 3): Promise<T> {
  let last: unknown;
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (err) {
      last = err;
      await new Promise((r) => setTimeout(r, 600 * (i + 1)));
    }
  }
  throw last;
}

interface ZipEntry {
  method: number;
  compSize: number;
  localOffset: number;
}

const zipDirs = new Map<string, Map<string, ZipEntry>>();

async function loadZipDir(url: string): Promise<Map<string, ZipEntry>> {
  const cached = zipDirs.get(url);
  if (cached) return cached;

  const tailRes = await fetchWithTimeout(url, { headers: { Range: "bytes=-131072" } });
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

  const dir = new Map<string, ZipEntry>();
  let p = 0;
  while (p + 46 <= cd.length) {
    if (cdv.getUint32(p, true) !== 0x02014b50) break;
    const method = cdv.getUint16(p + 10, true);
    const compSize = cdv.getUint32(p + 20, true);
    const nameLen = cdv.getUint16(p + 28, true);
    const extraLen = cdv.getUint16(p + 30, true);
    const commentLen = cdv.getUint16(p + 32, true);
    const localOffset = cdv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(cd.subarray(p + 46, p + 46 + nameLen));
    dir.set(name, { method, compSize, localOffset });
    p += 46 + nameLen + extraLen + commentLen;
  }
  zipDirs.set(url, dir);
  return dir;
}

async function extractTacozipEntry(url: string, entryName: string): Promise<Uint8Array> {
  const dir = await loadZipDir(url);
  const found = dir.get(entryName);
  if (!found) throw new Error(`Entry not found: ${entryName}`);

  const localHeader = await fetchRange(url, found.localOffset, found.localOffset + 29);
  const lv = new DataView(localHeader.buffer, localHeader.byteOffset, localHeader.byteLength);
  const dataStart = found.localOffset + 30 + lv.getUint16(26, true) + lv.getUint16(28, true);
  const raw = await fetchRange(url, dataStart, dataStart + found.compSize - 1);
  return found.method === 0 ? raw : await inflateRaw(raw);
}

async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function percentiles(data: ArrayLike<number>): { lo: number; hi: number } {
  const arr = Array.from(data).sort((a, b) => a - b);
  const at = (q: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.floor(arr.length * q)))];
  return { lo: at(0.02), hi: at(0.98) };
}

async function renderAsset(bytes: Uint8Array, asset: Asset): Promise<Rendered> {
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
        img.data[i * 4 + 3] = Math.round(Math.min(1, Math.max(0, (t - 0.06) / 0.22)) * 170);
      } else {
        img.data[i * 4] = 249;
        img.data[i * 4 + 1] = 115;
        img.data[i * 4 + 2] = 22;
        img.data[i * 4 + 3] = band[i] > 0 ? 210 : 0;
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

function polygonOf(coordinates: [number, number][]) {
  return {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [[...coordinates, coordinates[0]]] },
    properties: {},
  };
}

function popupHTML(p: SampleProps, asset?: Asset): string {
  const color = p.sensor === "EMIT" ? "#f97316" : p.sensor === "Sentinel-2" ? "#2dd4bf" : "#fde047";
  return (
    `<div class="pp"><span class="pp-sensor" style="color:${color}">${p.sensor}${asset ? ` (${ASSET_LABEL[asset]})` : ""}</span>` +
    `<h4>${p.country ?? "Unknown"}</h4>` +
    `<div class="pp-row"><span>Date</span><b>${p.date || "n/a"}</b></div>` +
    (p.flux ? `<div class="pp-row"><span>Flux</span><b>${Math.round(p.flux).toLocaleString()} kg/h</b></div>` : "") +
    `</div>`
  );
}

function attachHover(map: any, entry: Entry): void {
  const { fill } = ids(entry.key);
  if (bound.has(fill)) return;
  bound.add(fill);
  map.on("mouseenter", fill, (e: any) => {
    if (!popup) {
      popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10, className: "plume-popup", maxWidth: "280px" });
    }
    popup.setLngLat(e.lngLat).setHTML(popupHTML(entry.props, entry.asset)).addTo(map);
    map.getCanvas().style.cursor = "pointer";
  });
  map.on("mouseleave", fill, () => {
    if (popup) popup.remove();
    map.getCanvas().style.cursor = "";
  });
}

function addEntryLayers(map: any, entry: Entry): void {
  const { src, img, osrc, fill } = ids(entry.key);
  const vis = entry.visible ? "visible" : "none";
  if (!map.getSource(src)) {
    map.addSource(src, { type: "image", url: entry.rendered.dataUrl, coordinates: entry.rendered.coordinates });
    map.addSource(osrc, { type: "geojson", data: polygonOf(entry.rendered.coordinates) });
  }
  if (!map.getLayer(img)) {
    map.addLayer({ id: img, type: "raster", source: src, paint: { "raster-opacity": 0.95 }, layout: { visibility: vis } });
    map.addLayer({
      id: fill,
      type: "fill",
      source: osrc,
      paint: { "fill-color": "#000000", "fill-opacity": 0.01 },
      layout: { visibility: vis },
    });
  } else {
    map.setLayoutProperty(img, "visibility", vis);
    map.setLayoutProperty(fill, "visibility", vis);
  }
  attachHover(map, entry);
}

function removeEntryLayers(map: any, key: string): void {
  const { src, img, osrc, fill } = ids(key);
  for (const id of [img, fill]) if (map.getLayer(id)) map.removeLayer(id);
  for (const id of [src, osrc]) if (map.getSource(id)) map.removeSource(id);
}

function setVisible(entry: Entry, visible: boolean): void {
  entry.visible = visible;
  const { img, fill } = ids(entry.key);
  const vis = visible ? "visible" : "none";
  for (const id of [img, fill]) {
    if (mapRef.getLayer(id)) mapRef.setLayoutProperty(id, "visibility", vis);
  }
}

async function ensureLoaded(props: SampleProps, asset: Asset): Promise<Entry> {
  const key = keyOf(props.id, asset);
  const existing = entries.find((e) => e.key === key);
  if (existing) return existing;

  loading.add(key);
  failed.delete(key);
  render();
  try {
    const url = `${HF}/${props.dataset}/${props.file}`;
    let rendered = cache.get(key);
    if (!rendered) {
      const bytes = await withRetry(() => extractTacozipEntry(url, `DATA/${props.id}/${asset}`));
      rendered = await renderAsset(bytes, asset);
      cache.set(key, rendered);
    }
    const entry: Entry = { key, asset, props, rendered, visible: true };
    entries.push(entry);
    addEntryLayers(mapRef, entry);
    zoomToGroup([entry]);
    (window as any).__sampleLoaded = true;
    return entry;
  } catch (err) {
    failed.set(key, (err as Error).message);
    (window as any).__sampleError = (err as Error).message;
    throw err;
  } finally {
    loading.delete(key);
  }
}

function current(): Feature | undefined {
  return curList[curIndex];
}

function zoomToGroup(group: Entry[]): void {
  const coords = group.flatMap((e) => e.rendered.coordinates);
  const lngs = coords.map((c) => c[0]);
  const lats = coords.map((c) => c[1]);
  mapRef.fitBounds(
    [
      [Math.min(...lngs), Math.min(...lats)],
      [Math.max(...lngs), Math.max(...lats)],
    ],
    { padding: { top: 140, bottom: 140, left: 420, right: 340 }, duration: 800 },
  );
}

function renderLayersPanel(): void {
  if (!uiRef) return;
  const { layersRoot, layersBody } = uiRef;
  if (entries.length === 0) {
    layersRoot.style.display = "none";
    return;
  }
  layersRoot.style.display = "block";

  const groups = new Map<string, Entry[]>();
  for (const e of entries) {
    const g = groups.get(e.props.id) ?? [];
    g.push(e);
    groups.set(e.props.id, g);
  }

  layersBody.innerHTML =
    `<p class="globe-panel__title">On the map (${groups.size})</p>` +
    Array.from(groups.values())
      .map((group) => {
        const p = group[0].props;
        const chips = ASSETS.filter((a) => group.some((e) => e.asset === a))
          .map((a) => {
            const e = group.find((x) => x.asset === a)!;
            return `<button class="lp-chip ${e.visible ? "is-on" : ""}" data-chip="${e.key}" type="button">${ASSET_LABEL[a]}</button>`;
          })
          .join("");
        return (
          `<div class="lp-item">` +
          `<div class="lp-item__text"><b>${p.sensor}</b><span>${p.country}, ${p.date}</span></div>` +
          `<div class="lp-item__chips">${chips}</div>` +
          `<div class="lp-item__actions">` +
          `<button class="lp-icon" data-zoom="${p.id}" type="button" aria-label="Zoom">⌕</button>` +
          `<button class="lp-icon" data-remove="${p.id}" type="button" aria-label="Remove">×</button>` +
          `</div>` +
          `</div>`
        );
      })
      .join("");

  layersBody.querySelectorAll<HTMLButtonElement>("[data-chip]").forEach((el) => {
    el.addEventListener("click", () => {
      const entry = entries.find((e) => e.key === el.dataset.chip);
      if (!entry) return;
      setVisible(entry, !entry.visible);
      render();
    });
  });
  layersBody.querySelectorAll<HTMLButtonElement>("[data-zoom]").forEach((el) => {
    el.addEventListener("click", () => {
      const group = entries.filter((e) => e.props.id === el.dataset.zoom);
      if (group.length) zoomToGroup(group);
    });
  });
  layersBody.querySelectorAll<HTMLButtonElement>("[data-remove]").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.dataset.remove!;
      for (const e of entries.filter((x) => x.props.id === id)) {
        removeEntryLayers(mapRef, e.key);
        cache.delete(e.key);
      }
      for (let i = entries.length - 1; i >= 0; i--) {
        if (entries[i].props.id === id) entries.splice(i, 1);
      }
      render();
    });
  });
}

function render(): void {
  if (!uiRef) return;
  const ui = uiRef;
  const feature = current();
  renderLayersPanel();
  if (!feature) {
    ui.root.style.display = "none";
    return;
  }
  const p = feature.properties;
  ui.root.style.display = "block";

  const nav =
    curList.length > 1
      ? `<div class="sv-nav">` +
        `<button class="sv-nav__btn" data-move="-1" type="button" aria-label="Previous">‹</button>` +
        `<span class="sv-nav__pos">${curIndex + 1} / ${curList.length}</span>` +
        `<button class="sv-nav__btn" data-move="1" type="button" aria-label="Next">›</button>` +
        `</div>`
      : "";

  const head =
    `<p class="globe-panel__title">Sample</p>` +
    `<p class="sv-title">${p.sensor}, ${p.country ?? "Unknown"}</p>` +
    `<p class="sv-status">${p.date || "n/a"}, ${p.dataset}${p.flux ? `, ${Math.round(p.flux).toLocaleString()} kg/h` : ""}</p>`;

  let bodyHtml = "";
  if (p.viz === "multispectral") {
    bodyHtml =
      `<div class="sv-assets">` +
      ASSETS.map((a) => {
        const key = keyOf(p.id, a);
        const entry = entries.find((e) => e.key === key);
        const on = !!entry && entry.visible;
        const busy = loading.has(key);
        const err = failed.get(key);
        return (
          `<label class="sv-asset">` +
          `<input type="checkbox" data-asset="${a}" ${on ? "checked" : ""} ${busy ? "disabled" : ""}>` +
          `<span>${a === "target" ? "RGB" : a === "ch4" ? "CH₄ enhancement" : "Plume mask"}</span>` +
          `<span class="sv-asset__state${err ? " is-error" : ""}" title="${err ? err.replace(/"/g, "&quot;") : ""}">${busy ? "loading…" : entry ? (entry.visible ? "shown" : "hidden") : err ? "error" : ""}</span>` +
          `</label>`
        );
      }).join("") +
      `</div>`;
  } else {
    bodyHtml = `<p class="sv-status">EMIT is stored in sensor coordinates, so it cannot be drawn on the map yet.</p>`;
  }

  ui.body.innerHTML = nav + head + bodyHtml;

  ui.body.querySelectorAll<HTMLButtonElement>("[data-move]").forEach((el) => {
    el.addEventListener("click", () => move(Number(el.dataset.move)));
  });
  ui.body.querySelectorAll<HTMLInputElement>("[data-asset]").forEach((el) => {
    el.addEventListener("change", async () => {
      const asset = el.dataset.asset as Asset;
      const f = current();
      if (!f) return;
      try {
        if (el.checked) {
          const entry = await ensureLoaded(f.properties, asset);
          setVisible(entry, true);
        } else {
          const entry = entries.find((e) => e.key === keyOf(f.properties.id, asset));
          if (entry) setVisible(entry, false);
        }
      } catch (err) {
        (window as any).__sampleError = (err as Error).message;
        console.error(err);
      }
      render();
    });
  });
}

function move(delta: number): void {
  if (curList.length < 2) return;
  curIndex = (curIndex + delta + curList.length) % curList.length;
  const f = current();
  if (f) mapRef.easeTo({ center: f.geometry.coordinates, duration: 600 });
  render();
}

export function openInspector(map: any, feature: Feature, ui: UiElements): void {
  mapRef = map;
  uiRef = ui;
  const source = ui.list && ui.list.length ? ui.list : [feature];
  curList = [...source].sort((a, b) => (a.properties.date || "").localeCompare(b.properties.date || ""));
  curIndex = Math.max(0, curList.findIndex((f) => f.properties.id === feature.properties.id));
  render();
}

export function reAddAll(map: any): void {
  for (const entry of entries) {
    try {
      addEntryLayers(map, entry);
    } catch (e) {
      /* style not ready */
    }
  }
}
