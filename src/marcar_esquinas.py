#!/usr/bin/env python3
"""
Elegir el cuadro de referencia y marcar las 4 esquinas del arco para pipeline_balon.py.

Uso:
  python src/marcar_esquinas.py videos/penal1.mp4

Controles:
  barra "cuadro"  moverse por el video hasta un cuadro donde el arco se vea completo
  click           marcar esquinas INTERIORES: sup-izq, sup-der, inf-der, inf-izq
  z               deshacer el último click
  q / ESC         salir

Al marcar la 4ª esquina imprime el comando listo para copiar.
"""
import sys

import cv2

if len(sys.argv) < 2:
    sys.exit("Uso: python src/marcar_esquinas.py videos/video.mp4")
video = sys.argv[1]
cap = cv2.VideoCapture(video)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
nombres = ["superior izquierda", "superior derecha", "inferior derecha", "inferior izquierda"]
pts = []
estado = {"i": 0, "frame": None}


def leer(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    return f if ok else None


def cambiar(i):
    estado["i"] = i
    estado["frame"] = leer(i)
    pts.clear()


def click(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))
        if len(pts) == 4:
            esq = " ".join(f"{px},{py}" for px, py in pts)
            print(f"\n--ref {estado['i']} --esquinas {esq}\n")
            print("Comando:")
            print(f"python src/pipeline_balon.py {video} --ref {estado['i']} --esquinas {esq} "
                  f"--hasta {n_total - 3} --salida salida\n")


cv2.namedWindow("arco", cv2.WINDOW_NORMAL)
cv2.createTrackbar("cuadro", "arco", 0, max(1, n_total - 1), cambiar)
cv2.setMouseCallback("arco", click)
cambiar(0)
print(f"{n_total} cuadros a {fps:.0f} fps. Busca un cuadro con el arco completo y marca las esquinas.")

while True:
    f = estado["frame"]
    if f is None:
        break
    vis = f.copy()
    for k, p in enumerate(pts):
        cv2.circle(vis, p, 4, (0, 0, 255), -1)
        cv2.putText(vis, str(k + 1), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if len(pts) > 1:
        for a, b in zip(pts, pts[1:] + (pts[:1] if len(pts) == 4 else [])):
            cv2.line(vis, a, b, (0, 255, 255), 1)
    txt = f"cuadro {estado['i']} ({estado['i'] / fps:.2f}s)  |  "
    txt += f"click: esquina {nombres[len(pts)]}" if len(pts) < 4 else "listo, revisa la terminal"
    cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow("arco", vis)
    k = cv2.waitKey(20) & 0xFF
    if k in (ord("q"), 27):
        break
    if k == ord("z") and pts:
        pts.pop()
cv2.destroyAllWindows()
