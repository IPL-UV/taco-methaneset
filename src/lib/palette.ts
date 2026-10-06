export function paletteColor(name: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
  return value || "#94a3b8";
}

export function sensorColor(sensor: string): string {
  const token = sensor === "EMIT" ? "emit" : sensor === "Sentinel-2" ? "s2" : sensor === "Landsat 8/9" ? "l89" : "slate";
  return paletteColor(token);
}
