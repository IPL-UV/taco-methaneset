import { BaseClient, BaseResponse, fromArrayBuffer, fromCustomClient } from "geotiff";
import maplibregl from "maplibre-gl";
import proj4 from "proj4";

const HF = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main";

export interface SampleProps {
  dataset: string;
  sensor: string;
  system?: string;
  country: string;
  date: string;
  id: string;
  file: string;
  flux?: number | null;
  viz?: string;
  point?: [number, number];
}

type Asset = "target" | "ch4" | "plume" | "radiance" | "imeo" | "cm";

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
  band?: ArrayLike<number>;
  range?: { lo: number; hi: number };
  readRect?: [number, number, number, number];
  scene?: [number, number][];
  fullSize?: [number, number];
  zoomRead?: (rect: [number, number, number, number]) => Promise<Rendered | null>;
}

interface Entry {
  key: string;
  asset: Asset;
  props: SampleProps;
  rendered: Rendered;
  visible: boolean;
}

const MULTI_ASSETS: Asset[] = ["target", "ch4", "plume"];
const EMIT_ASSETS: Asset[] = ["radiance", "imeo", "cm"];
const ASSET_LABEL: Record<Asset, string> = {
  target: "RGB",
  ch4: "CH₄",
  plume: "Mask",
  radiance: "RGB",
  imeo: "IME-O",
  cm: "CM",
};
const ASSET_LONG: Record<Asset, string> = {
  target: "RGB",
  ch4: "CH₄ enhancement",
  plume: "Plume mask",
  radiance: "RGB (radiance)",
  imeo: "IME-O plume mask",
  cm: "Carbon Mapper mask",
};
const LEAF_OF: Record<string, string> = {
  target: "target",
  ch4: "ch4",
  plume: "plume",
};
const EMIT_SET = new Set<Asset>(["radiance", "imeo", "cm"]);

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
let closed = false;
let assetOrder: Asset[] = ["plume", "ch4", "target"];
let emitOrder: Asset[] = ["cm", "imeo", "radiance"];
const orderFor = (asset: Asset): Asset[] => (EMIT_SET.has(asset) ? emitOrder : assetOrder);

function sortEntries(): void {
  const rank = new Map<Asset, number>();
  for (const list of [assetOrder, emitOrder]) {
    [...list].reverse().forEach((a, i) => rank.set(a, i));
  }
  const sampleIdx = new Map<string, number>();
  entries.forEach((e) => {
    if (!sampleIdx.has(e.props.id)) sampleIdx.set(e.props.id, sampleIdx.size);
  });
  entries.sort((a, b) => {
    const sa = sampleIdx.get(a.props.id)!;
    const sb = sampleIdx.get(b.props.id)!;
    if (sa !== sb) return sa - sb;
    return (rank.get(a.asset) ?? 0) - (rank.get(b.asset) ?? 0);
  });
}

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

let ch4Min = 0;
let ch4Max = 2000;
let ch4BoundsLo = -1000;
let ch4BoundsHi = 2000;

function updateCh4Bounds(): void {
  const ranges = entries.filter((e) => e.asset === "ch4" && e.rendered.range).map((e) => e.rendered.range!);
  if (ranges.length) {
    const lo = Math.floor(Math.min(...ranges.map((r) => r.lo)) / 10) * 10;
    const hi = Math.ceil(Math.max(...ranges.map((r) => r.hi)) / 10) * 10;
    ch4BoundsLo = Math.max(-10000, lo);
    ch4BoundsHi = Math.min(10000, hi);
  }
  ch4Min = Math.max(ch4BoundsLo, Math.min(ch4Min, ch4BoundsHi - 10));
  ch4Max = Math.min(ch4BoundsHi, Math.max(ch4Max, ch4BoundsLo + 10));
}

function drawCh4Into(img: ImageData, band: ArrayLike<number>, width: number, height: number): void {
  const span = Math.max(1, ch4Max - ch4Min);
  for (let i = 0; i < width * height; i++) {
    const v = band[i];
    if (!isFinite(v) || v === 0 || (v > 0 && v < ch4Min)) {
      img.data[i * 4 + 3] = 0;
      continue;
    }
    const t = Math.max(0, Math.min(1, (v - ch4Min) / span));
    const [r, g, b] = colormap(t);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
}

function updateCh4Labels(): void {
  if (!uiRef) return;
  const body = uiRef.body;
  const setVal = (sel: string, value: number) => {
    const el = body.querySelector<HTMLInputElement>(sel);
    if (el) {
      el.min = String(ch4BoundsLo);
      el.max = String(ch4BoundsHi);
      el.value = String(value);
    }
  };
  setVal("[data-ch4-min]", ch4Min);
  setVal("[data-ch4-max]", ch4Max);
  setVal("[data-ch4-min-num]", ch4Min);
  setVal("[data-ch4-max-num]", ch4Max);
  const span = Math.max(1, ch4BoundsHi - ch4BoundsLo);
  const a = ((ch4Min - ch4BoundsLo) / span) * 100;
  const b = ((ch4Max - ch4BoundsLo) / span) * 100;
  const fill = body.querySelector<HTMLElement>("[data-ch4-fill]");
  if (fill) {
    fill.style.left = `${a}%`;
    fill.style.width = `${Math.max(0, b - a)}%`;
  }
  const minEl = body.querySelector<HTMLElement>("[data-ch4-min]");
  const maxEl = body.querySelector<HTMLElement>("[data-ch4-max]");
  if (minEl && maxEl) {
    minEl.style.zIndex = a > b - 12 ? "5" : "3";
    maxEl.style.zIndex = "4";
  }
}

function redrawCh4(): void {
  for (const entry of entries) {
    if (entry.asset !== "ch4" || !entry.rendered.band) continue;
    const canvas = document.createElement("canvas");
    canvas.width = entry.rendered.width;
    canvas.height = entry.rendered.height;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(canvas.width, canvas.height);
    drawCh4Into(img, entry.rendered.band, canvas.width, canvas.height);
    ctx.putImageData(img, 0, 0);
    entry.rendered.dataUrl = canvas.toDataURL("image/png");
    const { src } = ids(entry.key);
    const source = mapRef.getSource(src) as any;
    if (source && typeof source.updateImage === "function") {
      source.updateImage({ url: entry.rendered.dataUrl });
    }
  }
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

function percentiles(data: ArrayLike<number>, dropFill = false): { lo: number; hi: number } {
  const step = data.length > 150000 ? Math.ceil(data.length / 150000) : 1;
  const arr: number[] = [];
  for (let i = 0; i < data.length; i += step) {
    const v = data[i];
    if (!isFinite(v)) continue;
    if (dropFill && v <= -9990) continue;
    arr.push(v);
  }
  if (!arr.length) return { lo: 0, hi: 1 };
  arr.sort((a, b) => a - b);
  const at = (q: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.floor(arr.length * q)))];
  return { lo: at(0.02), hi: at(0.98) };
}

function paintCanvas(width: number, height: number, paint: (img: ImageData) => void): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  paint(img);
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

const HEAD_BYTES = 4 << 20;
const SMALL_FILE = 16 << 20;

class BufferedResponse extends BaseResponse {
  constructor(
    private code: number,
    private data: ArrayBuffer,
    private start: number,
    private total: number,
  ) {
    super();
  }
  get ok(): boolean {
    return true;
  }
  get status(): number {
    return this.code;
  }
  getHeader(name: string): string | undefined {
    const key = name.toLowerCase();
    if (key === "content-range") {
      return `bytes ${this.start}-${this.start + this.data.byteLength - 1}/${this.total}`;
    }
    if (key === "content-type") return "application/octet-stream";
    return undefined;
  }
  async getData(): Promise<ArrayBuffer> {
    return this.data;
  }
}

/**
 * Remote client for the EMIT folder files. The radiance is 1.48 GB, but its TIFF
 * directory and its pyramids live at the two ends of the file, and geotiff.js
 * would otherwise issue hundreds of tiny ranged reads. This prefetches the head
 * (directory) and the tail (pyramids), serves those slices from memory and only
 * hits the network for anything in between.
 */
class PrefetchClient extends BaseClient {
  private chunks: { start: number; data: ArrayBuffer }[] = [];
  private total = 0;
  private ready: Promise<void>;
  private resolved: string | null = null;

  constructor(url: string) {
    super(url);
    this.ready = this.prefetch();
  }

  /**
   * Hugging Face answers each ranged read with a redirect to its CDN, which doubles
   * the round trips. The first response reveals the final URL; reuse it, and fall
   * back to the resolve URL if the signed link has expired.
   */
  private async fetchRange(headers: Record<string, string>, signal?: AbortSignal): Promise<Response> {
    const target = this.resolved ?? this.url;
    let response = await fetch(target, { headers, signal });
    if (response.ok || response.status === 206) {
      if (!this.resolved && response.url && response.url !== this.url) this.resolved = response.url;
      return response;
    }
    if (this.resolved) {
      this.resolved = null;
      response = await fetch(this.url, { headers, signal });
    }
    return response;
  }

  private async prefetch(): Promise<void> {
    try {
      const head = await this.fetchRange({ Range: `bytes=0-${HEAD_BYTES - 1}` });
      if (head.status !== 206) return;
      const cr = head.headers.get("content-range");
      const total = cr ? Number(cr.split("/")[1]) : NaN;
      if (!Number.isFinite(total) || total <= 0) return;
      this.total = total;
      const headData = await head.arrayBuffer();
      this.chunks.push({ start: 0, data: headData });
      if (this.total <= SMALL_FILE) {
        if (headData.byteLength < this.total) {
          const rest = await this.fetchRange({ Range: `bytes=${headData.byteLength}-${this.total - 1}` });
          if (rest.ok) this.chunks.push({ start: headData.byteLength, data: await rest.arrayBuffer() });
        }
      } else {
        const start = Math.max(HEAD_BYTES, this.total - (8 << 20));
        const tail = await this.fetchRange({ Range: `bytes=${start}-${this.total - 1}` });
        if (tail.ok) this.chunks.push({ start, data: await tail.arrayBuffer() });
      }
    } catch {
      /* on-demand requests still work */
    }
  }

  private buffered(start: number, length: number): ArrayBuffer | null {
    for (const chunk of this.chunks) {
      if (start >= chunk.start && start + length <= chunk.start + chunk.data.byteLength) {
        return chunk.data.slice(start - chunk.start, start - chunk.start + length);
      }
    }
    return null;
  }

  async request({ headers, signal }: { headers?: Record<string, string>; signal?: AbortSignal } = {}): Promise<BaseResponse> {
    await this.ready;
    const range = headers?.Range ?? headers?.range;
    if (range) {
      const match = /bytes=(\d+)-(\d+)/.exec(range);
      if (match) {
        const start = Number(match[1]);
        const length = Number(match[2]) - start + 1;
        const data = this.buffered(start, length);
        if (data) return new BufferedResponse(206, data, start, this.total || start + length);
      }
    }
    const response = await this.fetchRange(headers ?? {}, signal);
    return {
      ok: response.ok,
      status: response.status,
      getHeader: (name: string) => response.headers.get(name) ?? undefined,
      getData: () => response.arrayBuffer(),
    } as unknown as BaseResponse;
  }
}

async function emitTiff(url: string) {
  return fromCustomClient(new PrefetchClient(url), { blockSize: 65536, cacheSize: 256 });
}

type Corners = [number, number][];

function bilinear(corners: Corners, u: number, v: number): [number, number] {
  const [tl, tr, br, bl] = corners;
  const lon = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0];
  const lat = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1];
  return [lon, lat];
}

function invertBilinearRaw(corners: Corners, target: [number, number]): [number, number] | null {
  let u = 0.5;
  let v = 0.5;
  for (let i = 0; i < 12; i++) {
    const [lon, lat] = bilinear(corners, u, v);
    const e = 1e-4;
    const [lonU, latU] = bilinear(corners, u + e, v);
    const [lonV, latV] = bilinear(corners, u, v + e);
    const f0 = lon - target[0];
    const f1 = lat - target[1];
    const j00 = (lonU - lon) / e;
    const j01 = (lonV - lon) / e;
    const j10 = (latU - lat) / e;
    const j11 = (latV - lat) / e;
    const det = j00 * j11 - j01 * j10;
    if (Math.abs(det) < 1e-12) break;
    u -= (j11 * f0 - j01 * f1) / det;
    v -= (-j10 * f0 + j00 * f1) / det;
  }
  const [lon, lat] = bilinear(corners, u, v);
  const residual = Math.hypot(lon - target[0], lat - target[1]);
  if (!Number.isFinite(residual) || residual > 0.05) return null;
  return [u, v];
}

function invertBilinear(corners: Corners, target: [number, number]): [number, number] {
  const uv = invertBilinearRaw(corners, target);
  if (!uv || uv[0] < -0.05 || uv[0] > 1.05 || uv[1] < -0.05 || uv[1] > 1.05) {
    return [0.5, 0.5];
  }
  return [Math.max(0, Math.min(1, uv[0])), Math.max(0, Math.min(1, uv[1]))];
}

async function emitCoordinates(latlonUrl: string): Promise<[number, number][]> {
  const tiff = await emitTiff(latlonUrl);
  const image = await tiff.getImage(0);
  const w = image.getWidth();
  const h = image.getHeight();
  const corners: [number, number][] = [
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1],
  ];
  const out: [number, number][] = [];
  for (const [x, y] of corners) {
    const [band0, band1] = (await image.readRasters({ samples: [0, 1], window: [x, y, x + 1, y + 1] })) as unknown as ArrayLike<number>[];
    let lat = Number(band0[0]);
    let lon = Number(band1[0]);
    if (Math.abs(lat) > 90 && Math.abs(lon) <= 90) {
      [lat, lon] = [lon, lat];
    }
    if (!isFinite(lat) || !isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
      throw new Error("cannot georeference this EMIT sample");
    }
    out.push([lon, lat]);
  }
  return out;
}

function paintRgb(bands: ArrayLike<number>[], width: number, height: number): string {
  const scales = bands.map((band) => percentiles(band, true));
  return paintCanvas(width, height, (img) => {
    for (let i = 0; i < width * height; i++) {
      let fill = false;
      for (let c = 0; c < 3; c++) {
        const band = bands[Math.min(c, bands.length - 1)];
        const raw = band[i];
        if (!isFinite(raw) || raw <= -9990) {
          fill = true;
          break;
        }
        const s = scales[Math.min(c, scales.length - 1)];
        const v = s.hi === s.lo ? 0 : (raw - s.lo) / (s.hi - s.lo);
        img.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
      }
      img.data[i * 4 + 3] = fill ? 0 : 255;
    }
  });
}

async function renderEmitRadiance(base: string, latlon: string, point?: [number, number]): Promise<Rendered> {
  const tiff = await emitTiff(`${base}/radiance.tif`);
  const levels = await tiff.getImageCount();
  const full = await tiff.getImage(0);
  if (full.getSamplesPerPixel() < 36) throw new Error("unexpected EMIT radiance layout");
  const fullWidth = full.getWidth();
  const fullHeight = full.getHeight();
  const scene = await emitCoordinates(latlon);
  const ux = (x: number) => x / (fullWidth - 1);
  const vy = (y: number) => y / (fullHeight - 1);
  const rectCoordinates = (rect: [number, number, number, number]): Corners => [
    bilinear(scene, ux(rect[0]), vy(rect[1])),
    bilinear(scene, ux(rect[2] - 1), vy(rect[1])),
    bilinear(scene, ux(rect[2] - 1), vy(rect[3] - 1)),
    bilinear(scene, ux(rect[0]), vy(rect[3] - 1)),
  ];
  const readFullRect = async (rect: [number, number, number, number]): Promise<Rendered> => {
    const bands = (await full.readRasters({ samples: [35, 23, 12], window: rect })) as unknown as ArrayLike<number>[];
    const width = rect[2] - rect[0];
    const height = rect[3] - rect[1];
    return {
      dataUrl: paintRgb(bands, width, height),
      coordinates: rectCoordinates(rect),
      width,
      height,
      readRect: rect,
      scene,
      fullSize: [fullWidth, fullHeight],
      zoomRead,
    };
  };
  const zoomRead = (rect: [number, number, number, number]) => readFullRect(rect);
  if (levels > 1) {
    const image = await tiff.getImage(1);
    const width = image.getWidth();
    const height = image.getHeight();
    const bands = (await image.readRasters({ samples: [35, 23, 12] })) as unknown as ArrayLike<number>[];
    return {
      dataUrl: paintRgb(bands, width, height),
      coordinates: scene,
      width,
      height,
      scene,
      fullSize: [fullWidth, fullHeight],
      zoomRead,
    };
  }
  const [u, v] = point ? invertBilinear(scene, point) : [0.5, 0.5];
  const cx = Math.round(u * (fullWidth - 1));
  const cy = Math.round(v * (fullHeight - 1));
  const winW = Math.min(640, fullWidth);
  const winH = Math.min(640, fullHeight);
  const x0 = Math.max(0, Math.min(fullWidth - winW, cx - Math.floor(winW / 2)));
  const y0 = Math.max(0, Math.min(fullHeight - winH, cy - Math.floor(winH / 2)));
  return readFullRect([x0, y0, x0 + winW, y0 + winH]);
}

async function renderEmitMask(url: string, latlon: string, color: [number, number, number]): Promise<Rendered> {
  const tiff = await emitTiff(url);
  const image = await tiff.getImage(0);
  const width = image.getWidth();
  const height = image.getHeight();
  const mask = ((await image.readRasters({ samples: [0] })) as unknown as ArrayLike<number | bigint>[])[0];
  const dataUrl = paintCanvas(width, height, (img) => {
    for (let i = 0; i < width * height; i++) {
      if (Number(mask[i]) !== 0) {
        img.data[i * 4] = color[0];
        img.data[i * 4 + 1] = color[1];
        img.data[i * 4 + 2] = color[2];
        img.data[i * 4 + 3] = 210;
      }
    }
  });
  const coordinates = await emitCoordinates(latlon);
  return { dataUrl, coordinates, width, height };
}

async function renderEmitAsset(props: SampleProps, asset: Asset): Promise<Rendered> {
  const granule = props.id.split(":")[0];
  const base = `${HF}/${props.dataset}/DATA/${granule}`;
  const latlon = `${base}/latlon.tif`;
  if (asset === "radiance") return renderEmitRadiance(base, latlon, props.point);
  if (asset === "imeo") return renderEmitMask(`${base}/plume_imeo.tif`, latlon, [15, 118, 110]);
  if (asset === "cm") return renderEmitMask(`${base}/plume_cm.tif`, latlon, [142, 36, 170]);
  throw new Error(`unsupported EMIT asset: ${asset}`);
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
  let rawBand: ArrayLike<number> | undefined;
  let rawRange: { lo: number; hi: number } | undefined;

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
    if (asset === "ch4") {
      drawCh4Into(img, band, width, height);
      rawBand = band;
      const vals = Array.from(band).filter((v) => isFinite(v) && v !== 0);
      vals.sort((a, b) => a - b);
      const q = (p: number) => (vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * p))] : 0);
      rawRange = { lo: q(0.01), hi: q(0.99) };
    } else {
      for (let i = 0; i < width * height; i++) {
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

  return { dataUrl: canvas.toDataURL("image/png"), coordinates, width, height, band: rawBand, range: rawRange };
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
    let rendered = cache.get(key);
    if (!rendered) {
      if (props.dataset === "methaneset-emit") {
        rendered = await renderEmitAsset(props, asset);
      } else {
        const url = `${HF}/${props.dataset}/${props.file}`;
        const bytes = await withRetry(() => extractTacozipEntry(url, `DATA/${props.id}/${LEAF_OF[asset]}`));
        rendered = await renderAsset(bytes, asset);
      }
      cache.set(key, rendered);
    }
    const entry: Entry = { key, asset, props, rendered, visible: true };
    entries.push(entry);
    sortEntries();
    addEntryLayers(mapRef, entry);
    zoomToGroup([entry]);
    if (asset === "ch4" && rendered.range) {
      updateCh4Bounds();
      ch4Min = Math.min(Math.max(0, ch4BoundsLo), ch4BoundsHi - 10);
      ch4Max = Math.max(Math.min(2000, ch4BoundsHi), ch4BoundsLo + 10);
      updateCh4Labels();
      redrawCh4();
    }
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

function applyLayerOrder(): void {
  for (const entry of entries) {
    const { img, fill } = ids(entry.key);
    if (mapRef?.getLayer(img)) mapRef.moveLayer(img);
    if (mapRef?.getLayer(fill)) mapRef.moveLayer(fill);
  }
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
        const chips = [...MULTI_ASSETS, ...EMIT_ASSETS].filter((a) => group.some((e) => e.asset === a))
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
  if (!feature || closed) {
    ui.root.style.display = "none";
    return;
  }
  const p = feature.properties;
  ui.root.style.display = "block";

  const head =
    `<p class="globe-panel__title">Sample</p>` +
    `<button class="sv-close" type="button" aria-label="Close">×</button>` +
    `<p class="sv-title">${p.sensor}${p.system ? ` (${p.system})` : ""}, ${p.country ?? "Unknown"}</p>` +
    `<p class="sv-status">${p.source ? `${p.source}, ` : ""}${p.date || "n/a"}, ${p.dataset}${p.flux ? `, ${Math.round(p.flux).toLocaleString()} kg/h` : ""}</p>`;

  const sampleAssets = p.dataset === "methaneset-emit" ? emitOrder : assetOrder;
  const bodyHtml =
    `<div class="sv-assets">` +
    sampleAssets
      .map((a) => {
        const key = keyOf(p.id, a);
        const entry = entries.find((e) => e.key === key);
        const on = !!entry && entry.visible;
        const busy = loading.has(key);
        const err = failed.get(key);
        return (
          `<div class="sv-asset-row" draggable="true" data-asset-row="${a}">` +
          `<span class="sv-asset-row__handle" aria-hidden="true">⠿</span>` +
          `<label class="sv-asset">` +
          `<input type="checkbox" data-asset="${a}" ${on ? "checked" : ""} ${busy ? "disabled" : ""}>` +
          `<span>${ASSET_LONG[a]}</span>` +
          `<span class="sv-asset__state${err ? " is-error" : ""}" title="${err ? err.replace(/"/g, "&quot;") : ""}">${busy ? "loading…" : entry ? (entry.visible ? "shown" : "hidden") : err ? "error" : ""}</span>` +
          `</label>` +
          `</div>`
        );
      })
      .join("") +
    `</div>`;

  updateCh4Bounds();
  const legend = entries.some((e) => e.asset === "ch4")
    ? `<div class="sv-legend">` +
      `<span class="sv-legend__label">ΔXCH₄ (ppb)</span>` +
      `<div class="sv-legend__bar"></div>` +
      `<div class="sv-legend__slider">` +
      `<div class="sv-legend__fill" data-ch4-fill></div>` +
      `<input type="range" min="${ch4BoundsLo}" max="${ch4BoundsHi}" step="10" value="${ch4Min}" data-ch4-min />` +
      `<input type="range" min="${ch4BoundsLo}" max="${ch4BoundsHi}" step="10" value="${ch4Max}" data-ch4-max />` +
      `</div>` +
      `<div class="sv-legend__inputs">` +
      `<input type="number" min="${ch4BoundsLo}" max="${ch4BoundsHi}" step="10" value="${ch4Min}" data-ch4-min-num />` +
      `<input type="number" min="${ch4BoundsLo}" max="${ch4BoundsHi}" step="10" value="${ch4Max}" data-ch4-max-num />` +
      `</div>` +
      `</div>`
    : "";

  const picks =
    curList.length > 1
      ? `<p class="sv-status">${curList.length} samples at this location</p>` +
        `<div class="sv-picks">` +
        curList
          .map((f, i) => {
            const q = f.properties;
            const meta = [q.date || "n/a", q.flux ? `${Math.round(q.flux).toLocaleString()} kg/h` : ""]
              .filter(Boolean)
              .join(", ");
            return (
              `<button class="sv-pick${i === curIndex ? " is-on" : ""}" data-pick="${i}" type="button">` +
              `<b>${q.sensor}${q.system ? ` (${q.system})` : ""}</b><span>${meta}</span>` +
              `</button>`
            );
          })
          .join("") +
        `</div>`
      : "";

  ui.body.innerHTML = head + picks + bodyHtml + legend;

  ui.body.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((el) => {
    el.addEventListener("click", () => {
      curIndex = Number(el.dataset.pick);
      render();
    });
  });
  ui.body.querySelector<HTMLButtonElement>(".sv-close")?.addEventListener("click", () => {
    closed = true;
    render();
  });
  const onCh4Min = (value: number) => {
    ch4Min = value;
    if (ch4Min > ch4Max - 10) ch4Max = Math.min(ch4BoundsHi, ch4Min + 10);
    updateCh4Labels();
    redrawCh4();
  };
  const onCh4Max = (value: number) => {
    ch4Max = value;
    if (ch4Max < ch4Min + 10) ch4Min = Math.max(ch4BoundsLo, ch4Max - 10);
    updateCh4Labels();
    redrawCh4();
  };
  ui.body.querySelector<HTMLInputElement>("[data-ch4-min]")?.addEventListener("input", (ev) => {
    onCh4Min(Number((ev.target as HTMLInputElement).value));
  });
  ui.body.querySelector<HTMLInputElement>("[data-ch4-max]")?.addEventListener("input", (ev) => {
    onCh4Max(Number((ev.target as HTMLInputElement).value));
  });
  ui.body.querySelector<HTMLInputElement>("[data-ch4-min-num]")?.addEventListener("change", (ev) => {
    onCh4Min(Number((ev.target as HTMLInputElement).value));
  });
  ui.body.querySelector<HTMLInputElement>("[data-ch4-max-num]")?.addEventListener("change", (ev) => {
    onCh4Max(Number((ev.target as HTMLInputElement).value));
  });
  updateCh4Labels();
  let dragAsset: Asset | null = null;
  ui.body.querySelectorAll<HTMLElement>("[data-asset-row]").forEach((row) => {
    row.addEventListener("dragstart", (ev) => {
      dragAsset = (row.dataset.assetRow as Asset) ?? null;
      row.classList.add("is-dragging");
      ev.dataTransfer?.setData("text/plain", dragAsset ?? "");
    });
    row.addEventListener("dragend", () => {
      dragAsset = null;
      row.classList.remove("is-dragging");
    });
    row.addEventListener("dragover", (ev) => ev.preventDefault());
    row.addEventListener("drop", (ev) => {
      ev.preventDefault();
      const target = row.dataset.assetRow as Asset;
      if (!dragAsset || dragAsset === target) return;
      const orderList = orderFor(dragAsset);
      const from = orderList.indexOf(dragAsset);
      const to = orderList.indexOf(target);
      if (from < 0 || to < 0) return;
      const [moved] = orderList.splice(from, 1);
      orderList.splice(to, 0, moved);
      sortEntries();
      applyLayerOrder();
      render();
    });
  });
  const selected = ui.body.querySelector<HTMLElement>(".sv-pick.is-on");
  if (selected) selected.scrollIntoView({ block: "nearest" });
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

export function openInspector(map: any, feature: Feature, ui: UiElements): void {
  mapRef = map;
  uiRef = ui;
  closed = false;
  const source = ui.list && ui.list.length ? ui.list : [feature];
  for (const f of source) {
    if (!f.properties.point) f.properties.point = f.geometry.coordinates;
  }
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
  try {
    applyLayerOrder();
  } catch (e) {
    /* style not ready */
  }
}

let zoomBound = false;
let zoomToken = 0;

export function attachEmitZoom(map: any): void {
  if (zoomBound) return;
  zoomBound = true;
  let timer: ReturnType<typeof setTimeout> | null = null;
  map.on("moveend", () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => void refreshEmitWindows(map), 700);
  });
}

function visibleRect(entry: Entry, map: any): [number, number, number, number] | null {
  const scene = entry.rendered.scene;
  const fullSize = entry.rendered.fullSize;
  if (!scene || !fullSize) return null;
  const bounds = map.getBounds();
  const pts: [number, number][] = [
    [bounds.getWest(), bounds.getNorth()],
    [bounds.getEast(), bounds.getNorth()],
    [bounds.getEast(), bounds.getSouth()],
    [bounds.getWest(), bounds.getSouth()],
  ];
  let minU = 1;
  let minV = 1;
  let maxU = 0;
  let maxV = 0;
  for (const p of pts) {
    const uv = invertBilinearRaw(scene, p);
    if (!uv) return null;
    minU = Math.min(minU, uv[0]);
    maxU = Math.max(maxU, uv[0]);
    minV = Math.min(minV, uv[1]);
    maxV = Math.max(maxV, uv[1]);
  }
  if (maxU < 0 || maxV < 0 || minU > 1 || minV > 1) return null;
  const [fw, fh] = fullSize;
  const mx = (maxU - minU) * 0.1;
  const my = (maxV - minV) * 0.1;
  const x0 = Math.max(0, Math.floor((minU - mx) * (fw - 1)));
  const y0 = Math.max(0, Math.floor((minV - my) * (fh - 1)));
  const x1 = Math.min(fw, Math.ceil((maxU + mx) * (fw - 1)) + 1);
  const y1 = Math.min(fh, Math.ceil((maxV + my) * (fh - 1)) + 1);
  if (x1 - x0 < 32 || y1 - y0 < 32) return null;
  if ((x1 - x0) * (y1 - y0) > 1024 * 1024) return null;
  return [x0, y0, x1, y1];
}

async function refreshEmitWindows(map: any): Promise<void> {
  if (map.getZoom() < 12) return;
  const token = ++zoomToken;
  for (const entry of entries) {
    if (entry.asset !== "radiance" || !entry.visible || !entry.rendered.zoomRead) continue;
    const rect = visibleRect(entry, map);
    if (!rect) continue;
    const loaded = entry.rendered.readRect;
    if (
      loaded &&
      rect[0] >= loaded[0] &&
      rect[1] >= loaded[1] &&
      rect[2] <= loaded[2] &&
      rect[3] <= loaded[3]
    ) {
      continue;
    }
    const zoomRead = entry.rendered.zoomRead;
    const result = await zoomRead(rect).catch(() => null);
    if (token !== zoomToken) return;
    if (!result) continue;
    entry.rendered = result;
    const { src } = ids(entry.key);
    const source = map.getSource(src) as any;
    if (source && typeof source.updateImage === "function") {
      source.updateImage({ url: result.dataUrl, coordinates: result.coordinates });
    }
  }
}
