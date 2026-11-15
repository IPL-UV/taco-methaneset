---
tags: [methanset, multispectral, calidad, plan]
status: en-curso
created: 2026-11-13
updated: 2026-11-14
repo: https://github.com/IPL-UV/taco-methaneset
---

# Reparación del finetune multispectral: auditoría y rebuild final

Objetivo: que cada referencia y cada columna de calidad esté sustentada y alineada con lo que
hace MARS, sin rehacer el TACO desde cero. Se repaqueta byte a byte y solo se tocan los píxeles
que están rotos.

## Auditoría completa (14 nov)

`v140/t13_auditoria.py` recorre estructura, coherencia de columnas, shape de `level1`, offsets,
unidades de los rásteres y recálculo de `bg:difference`. Resultado sobre el build 1.4.0 (T7),
1.370 L89 + 3.605 S2:

| # | Defecto | Alcance | Gravedad |
|---|---|---|---|
| D1 | `bg:sensor1..3` en S2 guarda el datatake (`20241205T100319`) y no el satélite | 3.552 muestras × 3 columnas | alta |
| D2 | `geotiff:stats` en L89 solo trae la banda 1 (S2 trae las 13) | 1.353 muestras × 5 hojas | media |
| D3 | Chips vacíos o casi vacíos: todo cero con 100 % válido | S2 34 target + 24 bg0 + 1 ch4; L89 2 refs con 2,5 % válido | alta |
| D4 | `level1` ordenado por entrada del zip (`ch4`, `pluma` antes de `bg1-3`) | 70 muestras (17 L89, 53 S2) | alta |
| D5 | `METADATA/level0.parquet` dentro del zip no incluye las 70 muestras nuevas | 13 zips | alta |
| D6 | Zip de Poland sin `METADATA` | 1 zip (1 muestra) | alta |
| D7 | `target:cloud` y `bg:cloud0-3` medidas a 10 m con modelos equivocados | las dos colecciones | alta |
| D8 | `bg:methane1-3` compara contra la mediana del pool: mide desfase de escena | las dos colecciones | media |
| D9 | `bg:difference` de S2 no coincide con el recálculo en 20 % de la muestra | por confirmar | por confirmar |
| D10 | Referencias por encima del 35 % de nube de MARS | 213 ranuras en 172 muestras | alta |

Comprobado y **correcto**: 62 columnas en `level0` (61 dentro del zip, la 62 es
`internal:source_file` del nivel de colección), 8 hojas por muestra, 11/13 bandas por hoja
salvo D3, offsets dentro del zip, sin ids duplicados, `target:sensor` coincide con el tile,
`bg:difference` de L89 coincide con el recálculo (desviación mediana 1e-7).

## Decisiones cerradas

- **`bg0` no se toca**: es la referencia de MARS (el par del paper). Las referencias nuestras
  son `bg1`-`bg3`.
- **MARS no comprueba metano en sus referencias**: `isplume` es código muerto en la ruta pública
  (`location_image.py:126`) y `query_only_images_without_plumes` viene en `False`. Nuestro
  criterio es nube + similitud, igual que ellos. La columna `bg:methane` se publica como
  diagnóstico y se documenta como lo que es.
- **Criterio de nube = el de MARS**: 5 % en la primera pasada, 35 % en la segunda. Se aplica por
  ranura (preferimos ≤5 % y solo rellenamos con ≤35 %).
- **No se tira ninguna muestra**: las hojas con calima fina tienen el suelo visible y MARS las
  aceptó. Solo se sustituyen referencias, nunca targets.
- **No se sube `dataset_version`**: se queda en 1.4.0.

## Plan de rebuild

0. **Auditoría cerrada** (`t13_auditoria.py`) con la tabla de arriba.
1. **Chips vacíos (D3)**: re-descargar las hojas rotas de GEE con la misma transformación y
   reescribir solo esas entradas del zip.
2. **Referencias (D7, D10)**: pool ampliado (`_v140/T12/raw/`), selección con CloudSEN12 nativo
   y dos pasadas; 172 muestras, 213 ranuras.
3. **Hojas nuevas**: montar `level0` y `level1` con `bg:sensor`, `bg:date`, `bg:cloud`,
   `bg:difference` y `bg:methane` recalculados; corregir D1 y D2 de paso.
4. **Repaquetado** (`t7_repack.py`): orden de `level1` por nombre de hoja (D4), `METADATA`
   interna completa (D5, D6), offsets y `TACO_HEADER` recomputados, píxeles intactos.
5. **Puertas**, todas obligatorias:
   - zips con hash idéntico si no les toca cambio;
   - dentro de cada zip: entradas DATA ajenas bit a bit iguales, offsets dentro del fichero,
     `METADATA` interna == `.tacocat` salvo `internal:source_file`;
   - `level0` con 62 columnas y los mismos dtypes;
   - `tacoreader.load()` local leyendo las 8 hojas de cada muestra reparada;
   - recuentos 1.370 + 3.605 = 4.975 y 4.820 con pluma;
   - `t13_auditoria.py` sin fallos en las dos colecciones.
6. **Publicación**: HF, cards (`t9_cards.py`), web y notas del vault. El paper no se toca.

## Estado al 14 nov, 15:30

- Auditoría completa hecha; seis defectos nuevos (D1-D6) además de los ya conocidos (D7-D10).
- Pool de los 54 sitios y de los 133 restantes descargado (`_v140/T12/raw/`), 2.660 chips.
- Selección de los 54 hecha: 162 referencias, 159 por debajo del 5 % de nube.
- Pipeline arreglado: `C02/T1_TOA` en el descargador, escala de `cloud_frac`, filtro de metano
  retirado (`--methane-max` por defecto en 1,0).
- Nada publicado se ha tocado.

## Cierre (14 nov, 21:00)

Rebuild terminado y publicado. Los detalles, cifras y hashes están en `CHANGELOG.md`; aquí
queda lo que importa para no repetirlo.

- **Build**: `_v140/T16/{l89,s2}` (1,73 GB y 6,22 GB). Sustituye al de T7.
- **Puertas**: `t20_puertas.py` con **0 fallos** en las dos colecciones: 10.811 + 28.593 hojas
  idénticas, 149 + 247 cambiadas (todas previstas), 0 cambiadas fuera de la lista, offsets
  contra los TIFF reales, `TACO_HEADER` contra el zip, `METADATA` == `DATA`, columnas y orden
  canónicos, nube ≤35 %, `COLLECTION.json` por zip y `tacoreader.load()`.
- **Auditoría**: `t13_auditoria.py` con **0 fallos y 0 avisos** en las dos colecciones.
- **Extras que aparecieron por el camino** (y quedaron arreglados): `COLLECTION.json` de cada
  zip con versión 1.2.0 y extensión global (D12), `bg:sensor0` de S2 con guiones bajos (D13),
  orden de columnas de S2 (D14), `bg:difference` NaN por el relleno -inf de Landsat (D15), dos
  referencias L89 con 2,5 % de cobertura que se sustituyeron por escenas completas.
- **Publicado**: HF `tacofoundation/methaneset` commit `cca1c94` (rama `rebuild-v140`
  verificada antes, con descarga de prueba fijada al commit).
- **Repetible**: `t16_rebuild.py` (escribe los zips), `t21_level0.py` (catálogos y
  `COLLECTION.json`), `t20_puertas.py` (puertas) y `t13_auditoria.py`. Cualquier cambio futuro
  entra por ahí y vuelve a pasar las mismas puertas; no hace falta rehacer el TACO a mano.
