const raw = import.meta.env.BASE_URL;

export const base = raw.endsWith("/") ? raw : `${raw}/`;

export const url = (path = "") => `${base}${path.replace(/^\/+/, "")}`;
