// EMIT viewer verification: per-catalog asset rows, pair/orphan flow, per-bit mask painting,
// Matches filter, paper palette per theme, layer z-order, in-flight dedupe and mobile layout.
//
// Usage:  NODE_PATH=/data/users/julio/slidev/node_modules /usr/bin/node scripts/verify_emit_viewer.cjs
// Requires the built dist/ (npm run build) at the repo root.
const http = require("http"), fs = require("fs"), path = require("path");
const { chromium } = require("playwright-chromium");
const ROOT = path.join(__dirname, "..", "dist"), PREFIX = "/taco-methaneset", PORT = 8091;
const GRAN = "EMIT_L1B_RAD_001_20220810T065132_2222205_041";
const ISR = "EMIT_L1B_RAD_001_20220810T064957_2222205_033:ISR_S_005";
const types = { ".html":"text/html",".js":"text/javascript",".css":"text/css",".svg":"image/svg+xml",".json":"application/json",".geojson":"application/json",".png":"image/png",".woff2":"font/woff2" };
const server = http.createServer((req,res)=>{let p=decodeURIComponent(req.url.split("?")[0]);if(p.startsWith(PREFIX))p=p.slice(PREFIX.length);if(p===""||p.endsWith("/"))p+="index.html";fs.readFile(path.join(ROOT,p),(e,d)=>{if(e){res.writeHead(404);res.end("nf");return;}res.writeHead(200,{"content-type":types[path.extname(p)]||"application/octet-stream"});res.end(d);});});

let failures = 0;
const check = (name, ok, extra = "") => {
  console.log(`${ok ? "ok" : "FAIL"}  ${name}${extra ? "  (" + extra + ")" : ""}`);
  if (!ok) failures++;
};

const rows = (pg) => pg.$$eval(".sv-asset-row", (rs) => rs.map((r) => ({
  asset: r.dataset.assetRow,
  label: r.querySelector(".sv-asset span:not(.sv-asset__dot):not(.sv-asset__state)")?.textContent,
  state: r.querySelector(".sv-asset__state")?.textContent,
})));
const chips = (pg) => pg.$$eval("[data-chip]", (cs) => cs.map((c) => ({ label: c.textContent, title: c.title })));
const clickPoint = async (pg, coords, zoom) => {
  await pg.evaluate(([c, z]) => window.__map.jumpTo({ center: c, zoom: z }), [coords, zoom]);
  await pg.waitForTimeout(2200);
  const pt = await pg.evaluate((c) => {
    const p = window.__map.project(c);
    const r = window.__map.getCanvas().getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  }, coords);
  await pg.mouse.click(pt.x, pt.y);
  await pg.waitForTimeout(900);
};
const settled = (pg) => pg.waitForFunction(() => !Array.from(document.querySelectorAll(".sv-asset__state")).some((s) => s.textContent === "loading…"), null, { timeout: 180000 });

(async () => {
  await new Promise((r) => server.listen(PORT, r));
  const b = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const pg = await b.newPage({ viewport: { width: 1500, height: 940 } });
  const errors = [];
  pg.on("pageerror", (e) => errors.push(String(e)));
  await pg.goto(`http://127.0.0.1:${PORT}${PREFIX}/data/`, { waitUntil: "load" });
  await pg.waitForFunction(() => window.__map && window.__map.getLayer && window.__map.getLayer("plumes"), null, { timeout: 60000 });
  await pg.waitForTimeout(2000);

  const counts = await pg.$$eval("[data-count-match]", (els) => Object.fromEntries(els.map((e) => [e.dataset.countMatch, e.textContent])));
  check("Matches counts 2826/1169", counts.pair === "2826" && counts.orphan === "1169", JSON.stringify(counts));
  const all = await pg.$eval("#globe-count", (e) => e.textContent);
  await pg.evaluate(() => { const b = document.querySelector('[data-match][value="orphan"]'); b.checked = false; b.dispatchEvent(new Event("change")); });
  await pg.waitForTimeout(400);
  const sin = await pg.$eval("#globe-count", (e) => e.textContent);
  check("Orphans filter removes 1169", all.startsWith("8900") && sin.startsWith("7731"), `${all} -> ${sin}`);
  await pg.evaluate(() => { const b = document.querySelector('[data-match][value="orphan"]'); b.checked = true; b.dispatchEvent(new Event("change")); });

  check("--emit dark token", (await pg.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--emit").trim())) === "#818cf8");
  await pg.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
  await pg.waitForTimeout(800);
  const light = await pg.evaluate(() => ({
    token: getComputedStyle(document.documentElement).getPropertyValue("--emit").trim(),
    paint: JSON.stringify(window.__map.getPaintProperty("plumes", "circle-color")),
  }));
  check("--emit light token and repaint", light.token === "#3730a3" && light.paint.includes("#3730a3"), light.token);
  await pg.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await pg.waitForTimeout(600);

  const gj = await pg.evaluate(async () => (await (await fetch("/taco-methaneset/data/plumes.geojson")).json()).features.map((f) => ({
    id: f.properties.id, source: f.properties.source, bit: f.properties.bit, match: f.properties.match,
    system: f.properties.system, pair_id: f.properties.pair_id, coords: f.geometry.coordinates,
  })));
  const isr = gj.find((f) => f.id === ISR);

  await clickPoint(pg, isr.coords, 9);
  const picks = await pg.$$(".sv-pick");
  if (picks.length) await picks[0].click();
  let r = await rows(pg);
  check("pair: rows listed but not loaded on click", r.length === 2 && r[0].asset === "imeo" && r[0].state === "" && r[1].asset === "radiance" && r[1].state === "", JSON.stringify(r));
  await pg.check('[data-asset="imeo"]');
  await settled(pg);
  await pg.check('[data-asset="radiance"]');
  await settled(pg);
  r = await rows(pg);
  check("pair: checkboxes load the layers", r[0].state === "shown" && r[1].state === "shown", JSON.stringify(r));
  const lay = await pg.evaluate(() => {
    const sv = document.getElementById("sample-viewer");
    const lp = document.getElementById("layer-panel");
    return { maxH: sv.style.maxHeight, gap: Math.round(lp.getBoundingClientRect().top - sv.getBoundingClientRect().bottom), lpDisplay: lp.style.display };
  });
  check("desktop: inspector capped above the layers panel", lay.maxH !== "" && lay.lpDisplay === "block" && lay.gap >= 0, JSON.stringify(lay));
  check("pair: Show button", (await pg.$eval("[data-pair]", (e) => e.textContent)).includes("Show"));
  await pg.click("[data-pair]");
  await pg.waitForFunction(() => Array.from(document.querySelectorAll(".sv-asset-row")).some((x) => x.textContent.includes("CM plume (pair)")) && !Array.from(document.querySelectorAll(".sv-asset__state")).some((s) => s.textContent === "loading…"), null, { timeout: 180000 });
  r = await rows(pg);
  check("pair: (pair) row between own and RGB", r.length === 3 && r[0].asset === "imeo" && r[1].asset === "cm" && r[1].label === "CM plume (pair)" && r[1].state === "shown" && r[2].asset === "radiance", JSON.stringify(r));
  check("pair: Hide button", (await pg.$eval("[data-pair]", (e) => e.textContent)).includes("Hide"));
  const cs = await chips(pg);
  check("chips: CM (pair) with mate id title", cs.some((c) => c.label === "CM (pair)" && c.title === isr.pair_id), JSON.stringify(cs));
  check("chips: own IMEO chip titled with plume id", cs.some((c) => c.label === "IMEO" && c.title === isr.source));
  const z = await pg.evaluate(() => {
    const layers = window.__map.getStyle().layers.map((l) => l.id);
    const own = layers.findIndex((i) => i.includes("2222205-033-imeo-ISR-S-005"));
    const mate = layers.findIndex((i) => i.includes("2222205-033-cm-CH4-6A"));
    const rgb = layers.findIndex((i) => i.includes("2222205-033-radiance"));
    return { own, mate, rgb };
  });
  check("z-order: own > pair > RGB", z.own > z.mate && z.mate > z.rgb && z.rgb >= 0, JSON.stringify(z));
  await pg.click("[data-pair]");
  await pg.waitForTimeout(400);
  r = await rows(pg);
  check("pair: Hide hides the row", r[1].state === "hidden");

  const orphan = gj.find((f) => f.match === "orphan" && f.system === "IMEO" && gj.filter((x) => x.coords.join() === f.coords.join()).length === 1);
  await clickPoint(pg, orphan.coords, 12);
  r = await rows(pg);
  check("orphan: own mask + RGB, no button, not loaded", r.length === 2 && r[0].asset === "imeo" && r[0].state === "" && !(await pg.$("[data-pair]")), JSON.stringify(r));
  check("orphan: text", (await pg.$eval(".sv-pair", (e) => e.textContent)).includes("Orphan: only in the IMEO catalog"));
  await pg.check('[data-asset="imeo"]');
  await settled(pg);
  check("orphan: mask loads on demand", (await rows(pg))[0].state === "shown");

  await pg.evaluate(async (gran) => {
    const data = await (await fetch("/taco-methaneset/data/plumes.geojson")).json();
    window.__map.getSource("plumes").setData({ type: "FeatureCollection", features: data.features.filter((f) => f.properties.id.startsWith(gran)) });
  }, GRAN);
  await pg.waitForTimeout(600);
  const plumes = gj.filter((f) => f.id.startsWith(GRAN) && f.system === "IMEO");
  const expected = { 1: 37, 2: 1151, 3: 221, 4: 523 };
  for (const plume of plumes) {
    await clickPoint(pg, plume.coords, 13.5);
    await pg.check('[data-asset="imeo"]');
    await settled(pg);
    const stats = await pg.evaluate(() => window.__maskStats ?? {});
    check(`bit ${plume.bit} (${plume.source}) paints ${expected[plume.bit]} px`, stats[`${GRAN}:imeo:${plume.bit}`] === expected[plume.bit], String(stats[`${GRAN}:imeo:${plume.bit}`]));
  }
  const loadedGranules = await pg.evaluate(() => Array.from(document.querySelectorAll("[data-chip]")).map((c) => (c.dataset.chip || "").split(":")[0]));
  const other = gj.find((f) => f.system === "IMEO" && f.match && !loadedGranules.includes(f.id.split(":")[0]) && gj.filter((x) => x.id.split(":")[0] === f.id.split(":")[0] && x.system === "IMEO").length >= 2);
  await pg.evaluate(async (gran) => {
    const data = await (await fetch("/taco-methaneset/data/plumes.geojson")).json();
    window.__map.getSource("plumes").setData({ type: "FeatureCollection", features: data.features.filter((f) => f.properties.id.startsWith(gran)) });
  }, other.id.split(":")[0]);
  await pg.waitForTimeout(600);
  const otherPlumes = gj.filter((f) => f.id.split(":")[0] === other.id.split(":")[0] && f.system === "IMEO");
  const chipsOfGroup = (granule) => pg.evaluate((gran) => {
    const group = Array.from(document.querySelectorAll(".lp-item")).find((el) => Array.from(el.querySelectorAll("[data-chip]")).some((c) => (c.dataset.chip || "").includes(gran)));
    return group ? Array.from(group.querySelectorAll("[data-chip]")).map((c) => c.textContent + "|" + c.title) : [];
  }, granule);
  const before = await chipsOfGroup(other.id.split(":")[0]);
  await clickPoint(pg, otherPlumes[0].coords, 13.5);
  await pg.check('[data-asset="radiance"]');
  await settled(pg);
  await clickPoint(pg, otherPlumes[1].coords, 13.5);
  const shared = await rows(pg);
  check("two plumes of one granule share the RGB entry", shared.find((x) => x.asset === "radiance")?.state === "shown", JSON.stringify(shared));
  const after = await chipsOfGroup(other.id.split(":")[0]);
  check(
    "two plumes of one granule: a single RGB entry",
    after.filter((l) => l.startsWith("RGB")).length === 1,
    JSON.stringify({ before, after }),
  );

  await pg.close();
  const mobile = await b.newPage({ viewport: { width: 390, height: 844 } });
  mobile.on("pageerror", (e) => errors.push("mobile: " + String(e)));
  await mobile.goto(`http://127.0.0.1:${PORT}${PREFIX}/data/`, { waitUntil: "load" });
  await mobile.waitForFunction(() => window.__map && window.__map.getLayer && window.__map.getLayer("plumes"), null, { timeout: 60000 });
  await mobile.waitForTimeout(1500);
  await mobile.evaluate(() => document.getElementById("globe").scrollIntoView({ block: "center" }));
  await mobile.waitForTimeout(600);
  const target = await mobile.evaluate((c) => {
    window.__map.jumpTo({ center: c, zoom: 9 });
    const p = window.__map.project(c);
    const canvas = window.__map.getCanvas();
    const r = canvas.getBoundingClientRect();
    window.__map.panBy([p.x - r.width * 0.5, p.y - r.height * 0.55], { duration: 0 });
    return { x: r.left + r.width * 0.5, y: r.top + r.height * 0.55 };
  }, isr.coords);
  await mobile.waitForTimeout(2200);
  const onCanvas = await mobile.evaluate((t) => document.elementFromPoint(t.x, t.y)?.classList.contains("maplibregl-canvas") ?? false, target);
  check("mobile: click lands on the map canvas", onCanvas);
  await mobile.mouse.click(target.x, target.y);
  await mobile.waitForTimeout(3000);
  const opened = await mobile.evaluate(() => ({
    display: document.getElementById("sample-viewer")?.style.display ?? "",
    rows: document.querySelectorAll(".sv-asset-row").length,
  }));
  check("mobile: inspector opens", opened.display === "block" && opened.rows >= 2, JSON.stringify(opened));
  const heights = [];
  for (let i = 0; i < 6; i++) {
    heights.push(await mobile.evaluate(() => document.getElementById("sample-viewer")?.style.maxHeight ?? ""));
    await mobile.evaluate(() => { const cb = document.querySelector('[data-asset="imeo"]'); if (cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event("change")); } });
    await mobile.waitForTimeout(300);
  }
  check("mobile: max-height stays empty", heights.every((h) => h === ""), JSON.stringify(heights));

  check("zero page errors", errors.length === 0, errors.slice(0, 2).join(" | "));
  await b.close();
  server.close();
  console.log(failures === 0 ? "ALL OK" : `${failures} FAILURES`);
  process.exit(failures === 0 ? 0 : 1);
})();
