---
tags: [methanset, multispectral, changelog, datos]
status: cerrado
created: 2026-11-14
updated: 2026-11-14
repo: https://github.com/IPL-UV/taco-methaneset
---

# Changelog del finetune multispectral (v1.4.0, rebuild del 14 nov 2026)

Sustituye por completo al build del 13 de noviembre. Los `.tacozip` se escribieron desde cero
(cabecera, offsets y `__meta__` recalculados) y cada hoja que no debía cambiar se copió byte a
byte: 39,404 hojas idénticas, 337 sustituidas y 0 cambiadas fuera de la lista.

## Qué cambia

| # | Defecto del build anterior | Arreglo |
|---|---|---|
| D3 | Chips vacíos (todo cero con 100 % válido): 35 target y 24 bg0 de S2, y 2 referencias L89 con 2,5 % de píxeles válidos | 59 chips S2 re-descargados + 2 referencias L89 sustituidas por escenas que cubren el chip (cobertura 1,000) |
| D7 | Nube medida a 10 m con modelos equivocados (OmniCloudMask) | CloudSEN12 a resolución nativa (30 m Landsat, 20 m S2) sobre cada hoja publicada |
| D8 | `bg:methane` medido contra la mediana del pool | recalculado contra la mediana de la propia hoja; se publica como diagnóstico |
| D10 | Referencias por encima del 35 % de nube | dos pasadas de MARS (5 % estricto, 35 % si faltan candidatas): 98.6 % de las hojas L89 y 99.1 % de las S2 están a ≤5 %; máximo 31.8 % y 34.4 % |
| D11 | Referencias fuera de la ventana de ±120 días de MARS | todas dentro de ±120 días en las dos colecciones |
| D12 | `COLLECTION.json` de cada zip con `dataset_version` 1.2.0, extensión global (-180..180) y texto de Landsat en S2 | regenerado por zip con la identidad de la colección (1.4.0) y extensión real del zip |
| D13 | `bg:sensor0` de S2 con guiones bajos sueltos (`S2B_`, 53 muestras) | normalizado (`S2A`/`S2B`); `bg:sensor1..3` = familia `S2` |
| D14 | `level0` de S2 con el orden de columnas de la v1.3 | orden canónico, idéntico al de L89 |
| D15 | `bg:difference` NaN en 17 hojas L89 (el relleno de Landsat es -inf, no NaN) | se compara solo donde ambos rásteres tienen dato; 0 NaN en las dos colecciones |
| D16 | `target:cloud` y `bg:cloud0` de los chips reparados se habían medido sobre píxeles en cero (0,0 y NaN) | nube recalculada sobre el chip publicado (`t24_nube_chips.py`); un target de S2 queda con 80 %, que es lo que tiene |
| D4-D6 | `level1` desordenado, `METADATA` interno incompleto y el zip de Poland sin `METADATA` | `level1` canónico, `METADATA` interno completo y Poland con `METADATA` y `COLLECTION.json` |

## Cifras finales

| | L89 | S2 |
|---|---:|---:|
| muestras | 1,370 | 3,605 |
| hojas de referencia | 4,110 | 10,815 |
| referencias a ≤5 % de nube | 98.6 % | 99.1 % |
| nube máxima de una referencia | 31.8 % | 34.4 % |
| referencias con `bg:methane` ≤ 0,03 | 85.0 % | 85.9 % |
| sustituciones de referencia | 149 | 188 |
| chips reparados | 2 | 59 |

## Puertas y auditoría

- `t20_puertas.py`: 0 fallos en las dos colecciones. Comprueba sha256 hoja por hoja, offsets
  contra los TIFF reales, `TACO_HEADER` contra el zip, `METADATA` contra `DATA`, columnas y
  orden canónico, nube ≤35 %, `COLLECTION.json` por zip y `tacoreader.load()`.
- `t13_auditoria.py`: 0 fallos y 0 avisos en las dos colecciones. Recalcula nube, diferencia y
  metano sobre las hojas publicadas y compara con las columnas.
- Aviso aceptado: 9 hojas L89 con cobertura parcial de swath entre 75 % y 99 % (los bordes de
  Landsat). No hay ninguna por debajo del 50 %.

## Publicación

- Rama de verificación `rebuild-v140` en `tacofoundation/methaneset`, con el manifiesto
  comparado fichero a fichero contra el local y una descarga de prueba fijada al commit.
- `main`: commits [`2d4aee9`](https://huggingface.co/datasets/tacofoundation/methaneset/commit/2d4aee92ede5d363177f5f34f6bdebb8225df95e) (L89)
  y [`e046516`](https://huggingface.co/datasets/tacofoundation/methaneset/commit/e046516229fe489d36c7e38ba3dfebd820f092cf) (S2).
  El manifiesto de `main` coincide con el de la rama en las dos colecciones.
- `dataset_version` se queda en **1.4.0** por decisión de Julio; este changelog y los hashes
  son la trazabilidad de lo que cambió dentro de esa versión.
- Código: `IPL-UV/taco-methaneset` commit `0fea497` (diferencia con relleno y ventana de 120 días).

## Hashes (sha256 de cada `.tacozip` publicado)
### methaneset-l89-finetune (19 ficheros, 1.73 GB)

| fichero | bytes | sha256 |
|---|---:|---|
| `methaneset-l89-finetune_Algeria.tacozip` | 370420963 | `a640ac2f215f47344bbbc031652142f81aa73b390a886527458619e7729d2672` |
| `methaneset-l89-finetune_Argentina.tacozip` | 1294175 | `43cd240807f0954d4df35c98a0edab3c78e5692094eac628015391c906cc5b7b` |
| `methaneset-l89-finetune_Australia.tacozip` | 1325550 | `f29b9ae7c6c8c10810e64774089180a93fcd7882ee6ce4da4eec9e60a7b8553f` |
| `methaneset-l89-finetune_Bahrain.tacozip` | 9719208 | `bc40a60f55b0336ca3aae463fc3354965c3f5c415aed3ab80eb3f1e1188c7d26` |
| `methaneset-l89-finetune_Egypt.tacozip` | 13184314 | `857e2bebc39597fa8c78324d53abcc6444e77df1cada48a3746542d2c6adb273` |
| `methaneset-l89-finetune_India.tacozip` | 2620626 | `15613b8581a02c344b4101abffb85098cec2bf8ca7df3c8d56d45412244b4014` |
| `methaneset-l89-finetune_Iran_(Islamic_Republic_of).tacozip` | 55228554 | `d807be2a4cd0fd2a6a2f368638a92ff32b1d756a067b7eebe33efebefe1892ea` |
| `methaneset-l89-finetune_Iraq.tacozip` | 34709510 | `17745ab8e7d85720331acb391dd119dc351154775e1be3612b1960421430ce31` |
| `methaneset-l89-finetune_Kazakhstan.tacozip` | 32609013 | `0d57a44ef1f6d421eabab81dcf5c94e2648120ad2104849e9f29613c2cefe104` |
| `methaneset-l89-finetune_Kuwait.tacozip` | 5422303 | `964de7db46e0b2376e434923a38289f68de47123d539ba0652a7d4a0a84556e9` |
| `methaneset-l89-finetune_Libya.tacozip` | 139955699 | `5023d48cc45f7795eda19a00704a89de7b208055e730e0e9e73c0fc771a5e697` |
| `methaneset-l89-finetune_Oman.tacozip` | 1308693 | `1f39e6dffa58563343762c27327038441a72e8b9340523396d64deddc976bc6c` |
| `methaneset-l89-finetune_Saudi_Arabia.tacozip` | 14596419 | `dc2392fd54e391760ea50861f8ac78cc5b38ee8cea11c23a22cc8af3910b71a7` |
| `methaneset-l89-finetune_Syrian_Arab_Republic.tacozip` | 41804522 | `c11ff7b8edbd292f9c5387f0117c6174391efebff84f0a4f62530212c46794db` |
| `methaneset-l89-finetune_Turkmenistan.tacozip` | 839024068 | `0f57327a4326533f2b1b37976d7a74489e7f85d647cfcd39067403f03013ac52` |
| `methaneset-l89-finetune_United_States_of_America.tacozip` | 107106529 | `cadb81c36c01ff05c493227d9a79b1c37380ebaacb04b72c236b41db1ede1763` |
| `methaneset-l89-finetune_Uzbekistan.tacozip` | 9330238 | `28b26e2c4c97a95a78b604bc1ea11b7702a81a146ae1e4f088e5f263bdd0130a` |
| `methaneset-l89-finetune_Venezuela.tacozip` | 2898836 | `6a621278d931bc7b75f411b5b7d5ee60bfc410990bcf67b458e73c3ae49d87d1` |
| `methaneset-l89-finetune_Yemen.tacozip` | 44594805 | `ba8e71793b9c30fd34a1ca7ae2343524c869305093d50c7da598977c1a9644cf` |

### methaneset-s2-finetune (21 ficheros, 6.22 GB)

| fichero | bytes | sha256 |
|---|---:|---|
| `methaneset-s2-finetune_Algeria.tacozip` | 1791821132 | `68e65fa50db46c6422feca562b966199e687d6065978db4b61662e4ac67255c8` |
| `methaneset-s2-finetune_Argentina.tacozip` | 5286768 | `f531899ace4ef87b6ceeaa19edff4dc795b2f6f4eaaad53bd38da46efa085fa3` |
| `methaneset-s2-finetune_Bahrain.tacozip` | 37845061 | `424ac6a94a858f69456ecc3d76ab3e87dcb9cc3a9e6fc25c7e2f4901e157f55b` |
| `methaneset-s2-finetune_China.tacozip` | 5454581 | `c3e09987a8fb60785a7c7e788fa7f31ada1d7d9ec246bbfc229ead134fe4e9de` |
| `methaneset-s2-finetune_Egypt.tacozip` | 42876524 | `c205b177d4cd05e46400b44cd6e89823bfd4c2529fcf6382a83d0db5126b4d5d` |
| `methaneset-s2-finetune_Iran_(Islamic_Republic_of).tacozip` | 86223836 | `1d8199b8f7c27dc174425050f5b894e29e4eb5c645d8293aa2564493cebf2927` |
| `methaneset-s2-finetune_Iraq.tacozip` | 57808878 | `f1542563528fa0f1c9fba489474375e3158f36865bf45da912179cd992988a5e` |
| `methaneset-s2-finetune_Jordan.tacozip` | 3548348 | `9856482cf56ddb3345f58532b961238f17c5d5b12a1f92963188058b0e951c80` |
| `methaneset-s2-finetune_Kazakhstan.tacozip` | 478962152 | `4e1edccf8031f808e7ac9d5b0a1f4837b3e48f611dc09a44a8812d5390c9bb2e` |
| `methaneset-s2-finetune_Kuwait.tacozip` | 31864070 | `379d5263f10faf2720881ab4fff03b98ac7c8d8a2c3b6f35adf37228a4377a6b` |
| `methaneset-s2-finetune_Libya.tacozip` | 265312459 | `f4be60d05298c55fddd9554de7ba433b50fbf37eebded8a4d80be2e2b2e2b55b` |
| `methaneset-s2-finetune_Oman.tacozip` | 3499539 | `cc688a251f6d096efd2fd67cdd2dbe2268c671ba478945bd75c03e5833f0b3cd` |
| `methaneset-s2-finetune_Poland.tacozip` | 2459814 | `0ac8c62641eff9815e678d8b70c6503b15ebd4ba8c8c490ce04d3845aba91b57` |
| `methaneset-s2-finetune_Russian_Federation.tacozip` | 1683350 | `24225beffd477ba4196ce82d76220e2a0080524a338a0d6ac95f9ffb951b9291` |
| `methaneset-s2-finetune_Saudi_Arabia.tacozip` | 35857958 | `d0fb8e1a2b05f87278a86b9271bd1118bdc5aab952e43ae7a2d6906c398afe95` |
| `methaneset-s2-finetune_Syrian_Arab_Republic.tacozip` | 87553811 | `cf6cac4734e0016c08f97dca613744152ea0cbfb48850dbe35d9ab6c2a9ee831` |
| `methaneset-s2-finetune_Turkmenistan.tacozip` | 2844310786 | `49c464dcf7e76692cde655d7bbd0652b3ce465bde9ce38dfc9217e740de416de` |
| `methaneset-s2-finetune_United_States_of_America.tacozip` | 210821809 | `5e0686b20b12b1b07bfbdf1a8a63a10f6cc879766f0e5381dafc987df9bb144c` |
| `methaneset-s2-finetune_Uzbekistan.tacozip` | 20350502 | `734869862153c279f92fafbaebac0ec5705c4a4e8b59e250ae49142c9fd934a3` |
| `methaneset-s2-finetune_Venezuela.tacozip` | 3722127 | `5ed0ed417ed82390db2dfbcdf820467a61236196f0ce4e5890c0499a63f55b8e` |
| `methaneset-s2-finetune_Yemen.tacozip` | 200381985 | `6bac97e783ec1f99b908626f54265367042ddc99a857a06e62b998d404829f0b` |
