# HandStats

Análisis automático de lanzamientos al arco en balonmano a partir de video.

Proyecto del ramo **TEL-328 Procesamiento Digital de Imágenes** (UTFSM).

La idea salió de una reunión con la Federación de Balonmano de Chile: tener una herramienta que, a partir del video de una cámara apuntando al arco, detecte cada lanzamiento y diga hacia qué zona del arco fue, si terminó en gol, si el arquero atajó (y dónde) y, si se puede, a qué velocidad iba el balón.

![Trayectoria del balón detectada](docs/trayectoria_balon.png)

## Estado actual

- [x] Estabilización de video grabado con cámara en mano (ORB + homografía con RANSAC).
- [x] Arco marcado una vez por video y grilla de 3x3 zonas mediante homografía al plano del arco (300 x 200 cm).
- [x] Herramienta para elegir el cuadro de referencia y marcar las esquinas del arco con clicks.
- [x] Detección del balón por diferencia de 3 cuadros, con filtro por tamaño y descarte de jugadores.
- [x] Seguimiento del balón con modelo de velocidad constante, desde la mano del lanzador hasta el arco.
- [x] Clasificación básica: gol cuando el balón entra al arco y la red lo frena.
- [ ] Fine-tuning de YOLO con un dataset propio.
- [ ] Distinguir atajada y registrar dónde ataja el arquero.
- [ ] Estimar la velocidad del lanzamiento.
- [ ] Estadísticas de partido completo (versión futura).

## Estructura

```
src/
  pipeline_balon.py   pipeline clásico: estabilización, detección y seguimiento del balón
  marcar_esquinas.py  elegir el cuadro de referencia y marcar las esquinas del arco con clicks
  demo_arco.py        versión para cámara fija frontal: YOLO o sustracción de fondo + zona del arco
FlujoDemo.sh          comandos usados en la demo
docs/                 resultados de ejemplo (video en cámara lenta, figura, CSV)
videos/               poner aquí los videos (no se suben al repo)
salida/               resultados generados (no se suben al repo)
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- `ultralytics` solo hace falta para `demo_arco.py --metodo yolo`.
- `ffmpeg` es opcional. Si está instalado, los videos de salida se guardan en H.264 (se reproducen en cualquier lado); si no, quedan en MPEG-4.

## Uso

### 1. Marcar el arco (marcar_esquinas.py)

```bash
python src/marcar_esquinas.py videos/penal1.mp4
```

Con la barra se busca un cuadro donde el arco se vea completo y sin que nadie tape las esquinas. Después se hace click en las 4 esquinas **interiores** del arco, en este orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda. Con `z` se deshace el último click. Al marcar la cuarta esquina, la terminal muestra el comando listo para copiar.

El `--hasta` que sugiere es el final del video. Si la cámara se mueve mucho al final y el arco sale de la imagen, conviene bajarlo al último cuadro donde todavía se vea el arco.

### 2. Procesar el video (pipeline_balon.py)

```bash
python src/pipeline_balon.py videos/penal1.mp4 --ref 180 \
    --esquinas 37,399 128,399 128,487 37,490 --hasta 330 --salida salida
```

| Parámetro | Qué es |
|---|---|
| `--ref` | cuadro de referencia, donde se marcaron las esquinas |
| `--esquinas` | esquinas interiores del arco en ese cuadro (x,y) |
| `--hasta` | último cuadro a procesar |
| `--salida` | carpeta de resultados (se crea si no existe) |

Los valores del ejemplo son para `penal1.mp4`. Para otro video hay que sacarlos de nuevo con `marcar_esquinas.py`.

Genera en `salida/`:

- `demo_balon.mp4`: video original con el arco seguido y la trayectoria del balón, junto a la máscara de movimiento.
- `demo_zoom_lento.mp4`: zoom desde el lanzador hasta el arco, en cámara lenta (x4).
- `trayectoria_balon.png`: figura con la trayectoria completa sobre el cuadro de referencia.
- `trayectoria_balon.csv`: posición del balón por cuadro.
- `resultado.txt`: resultado del lanzamiento y zona.

Para repetir desde cero se puede vaciar la carpeta de resultados:

```powershell
Remove-Item salida\* -Exclude .gitkeep      # Windows
```

### Cámara fija frontal al arco (demo_arco.py)

Pensado para el montaje definitivo: celular en trípode detrás del lanzador, mirando al arco, a 60 fps. No sirve para videos grabados en mano o de lado. Al partir pide hacer click en las 4 esquinas del arco.

```bash
python src/demo_arco.py videos/sesion.mp4                  # YOLO pre-entrenado (clase "sports ball")
python src/demo_arco.py videos/sesion.mp4 --metodo fondo   # sustracción de fondo
```

Genera, al lado del video, el video anotado, un CSV con cada lanzamiento y un mapa del arco con todos los tiros.

## Cómo funciona el pipeline

1. **Estabilización:** se buscan puntos ORB en la parte estática de la imagen (pared y arco) y se calcula una homografía de cada cuadro al de referencia. Con eso el arco queda fijo aunque la cámara se mueva.
2. **Arco:** con las 4 esquinas se calcula la homografía imagen → plano del arco, que se usa para dibujar la grilla y para convertir posiciones a centímetros.
3. **Balón:** se hace la diferencia entre los cuadros t-2, t y t+2 del video estabilizado y se combinan, lo que deja solo lo que se movió en el cuadro t. Después se aplica umbral, morfología y filtro por tamaño, y se descartan las manchas pegadas a objetos grandes (jugadores). La búsqueda se hace en todo el ancho de la imagen, a la altura del arco, porque el tiro viene desde la cancha.
4. **Seguimiento:** los candidatos se unen entre cuadros con predicción de velocidad constante. La pista del balón es la que tiene la racha más larga de pasos rápidos y en la misma dirección. Después se completa hacia atrás, con la misma dirección y velocidad, hasta la mano del lanzador.
5. **Resultado:** se busca el punto donde la trayectoria entra al arco. Si después de entrar el balón se frena (la red lo detiene), se marca como gol. Después del gol solo se consideran 0,4 s y los puntos dentro del arco, para no confundir el movimiento de la red con el balón.

## Limitaciones conocidas

- Con video comprimido (WhatsApp) el balón mide unos 5 píxeles, lo que está en el límite de lo detectable.
- Grabando de lado, la fila de la zona es confiable pero la columna no. Por eso el montaje definitivo es con cámara fija y frontal.
- Las esquinas del arco se marcan a mano una vez por video.
- Por ahora se analiza un lanzamiento por video.

## Equipo

- Matías Ramos
- Ignacio Cubillos
- Bruno Yáñez