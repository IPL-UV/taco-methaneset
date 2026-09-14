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

interface UiElements {
  root: HTMLElement;
  body: HTMLElement;
}

let lastSample: { dataUrl: string; coordinates: [number, number][] } | null = null;

export function redrawSample(map: any): void {
  if (!lastSample || map.getSource("sample")) return;
  try {
    map.addSource("sample", { type: "image", url: lastSample.dataUrl, coordinates: lastSample.coordinates });
    map.addLayer({ id: "sample", type: "raster", source: "sample", paint: { "raster-opacity": 0.9 } });
  } catch (e) {
    /* style not ready yet */
  }
}

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

  let cd: Uint8Array;
  if (cdOffset >= tailStart) {
    cd = tail.subarray(cdOffset - tailStart);
  } else {
    cd = await fetchRange(url, cdOffset, cdOffset + cdSize - 1);
  }
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
  const lNameLen = lv.getUint16(26, true);
  const lExtraLen = lv.getUint16(28, true);
  const dataStart = found.localOffset + 30 + lNameLen + lExtraLen;
  const raw = await fetchRange(url, dataStart, dataStart + found.compSize - 1);
  return found.method === 0 ? raw : await inflateRaw(raw);
}

function stretch(data: ArrayLike<number>, lo = 0.02, hi = 0.98): { min: number; max: number } {
  const arr = Array.from(data).sort((a, b) => a - b);
  return { min: arr[Math.floor(arr.length * lo)], max: arr[Math.floor(arr.length * hi)] };
}

async function renderRgb(tiffBytes: Uint8Array) {
  const buffer = tiffBytes.buffer.slice(
    tiffBytes.byteOffset,
    tiffBytes.byteOffset + tiffBytes.byteLength,
  ) as ArrayBuffer;
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const count = image.getSamplesPerPixel();
  const samples = count >= 13 ? [11, 7, 3] : count >= 11 ? [6, 4, 3] : [0];
  const rasters = (await image.readRasters({ samples })) as ArrayLike<number>[];
  const bands = rasters as unknown as ArrayLike<number>[];

  const scales = bands.map((b) => stretch(b));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) {
      const band = bands[Math.min(c, bands.length - 1)];
      const s = scales[Math.min(c, scales.length - 1)];
      const v = s.max === s.min ? 0 : (band[i] - s.min) / (s.max - s.min);
      img.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
    }
    img.data[i * 4 + 3] = 255;
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
  const lnglat = corners.map(([x, y]) => {
    const [lng, lat] = proj ? proj4(proj, "EPSG:4326", [x, y]) : [x, y];
    if (!isFinite(lat) || Math.abs(lat) > 90 || Math.abs(lng) > 180) {
      throw new Error("cannot georeference this sample");
    }
    return [lng, lat] as [number, number];
  });

  return { dataUrl: canvas.toDataURL("image/png"), coordinates: lnglat, width, height };
}

export async function showSample(map: any, props: SampleProps, ui: UiElements): Promise<void> {
  const { root, body } = ui;
  root.style.display = "block";
  body.innerHTML = `<p class="sv-status">Loading sample · this reads a byte range from Hugging Face…</p>`;

  const url = `${HF}/${props.dataset}/${props.file}`;
  try {
    const bytes = await extractTacozipEntry(url, `DATA/${props.id}/target`);
    const { dataUrl, coordinates, width, height } = await renderRgb(bytes);
    lastSample = { dataUrl, coordinates };

    if (map.getLayer("sample")) map.removeLayer("sample");
    if (map.getSource("sample")) map.removeSource("sample");
    map.addSource("sample", { type: "image", url: dataUrl, coordinates });
    map.addLayer({ id: "sample", type: "raster", source: "sample", paint: { "raster-opacity": 0.9 } });

    body.innerHTML =
      `<p class="sv-title">${props.sensor} · ${props.country}</p>` +
      `<p class="sv-status">${props.date} · ${width}×${height} px · ${props.dataset}</p>` +
      `<button class="sv-clear" type="button">Clear</button>`;
    body.querySelector(".sv-clear")?.addEventListener("click", () => {
      if (map.getLayer("sample")) map.removeLayer("sample");
      if (map.getSource("sample")) map.removeSource("sample");
      root.style.display = "none";
    });

    (window as any).__sampleLoaded = true;
  } catch (err) {
    body.innerHTML = `<p class="sv-status sv-error">Could not load this sample. ${(err as Error).message}</p>`;
    (window as any).__sampleError = (err as Error).message;
    console.error(err);
  }
}
