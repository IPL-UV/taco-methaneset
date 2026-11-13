"""Reglas congeladas de la v1.4.0 del finetune multispectral.

Se importan desde los scripts de recuperacion. No se editan a mitad de la version: si algo
cambia, se sube de version y se anota en el plan.

Por que 1.4.0 y no 1.3.0: el arreglo de fondos de L89 de noviembre ya se llamo "Rebuild
v1.3.0" en las notas, aunque el `.tacocat` publicado siguiera en 1.2.0. Reusar el numero
mezcla dos cosas distintas.
"""

VERSION = "1.4.0"
LICENCIA = "CC-BY-NC-SA-4.0"

# Referencias: escalera documentada de MARS-S2L. La primera pasada es estricta y la segunda
# relaja; hoy el codigo usa 0,10 en las dos y por eso no relaja nada.
CLOUD_MAX = 0.50
CLOUD_MAX_RELAX = 0.50
METHANE_MAX = 0.03

# Ventana alrededor de la fecha del target. Se admite antes y despues, como en MARS-S2L.
DIAS_VENTANA = 120

# Cobertura minima de pixeles validos en la hoja descargada. Es la que ya usa 01_descargar.py.
COBERTURA_MIN = 0.90

# Un fondo que es el target de otra muestra trae la pluma de esa muestra. Nunca.
EXCLUIR_TARGETS_COMO_FONDO = True

# Mismo numero de referencias para todas: si un sitio no llega a 3, se registra el motivo y
# se queda fuera. Nada de hojas variables.
REFERENCIAS_POR_MUESTRA = 3

# Las 4.905 muestras publicadas se copian byte a byte del TACO actual. No se regeneran.
COPIAR_PUBLICADAS = True

# Las 141 offshore no tienen referencia en el catalogo: fuera, y documentado.
INCLUIR_OFFSHORE = False

# Las 372 sin split_name _2023 no entran en esta version: asignarles particion cambiaria la
# composicion del split y eso es una decision aparte.
INCLUIR_SIN_SPLIT = False
