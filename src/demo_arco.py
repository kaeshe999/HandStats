#!/usr/bin/env python3
"""
TEL-328 · Demo de avance: detección del balón y zona del arco (balonmano).

Uso:
  python demo_arco.py video.mp4                   # YOLO pre-entrenado (COCO 'sports ball')
  python demo_arco.py video.mp4 --metodo fondo    # sustracción de fondo (cámara fija)
  python demo_arco.py video.mp4 --recalibrar      # volver a marcar las esquinas del arco
  python demo_arco.py video.mp4 --sin-ventana     # sin vista previa (más rápido)

Salidas (junto al video):
  <video>_anotado.mp4       video con arco, grilla 3x3, detecciones y trayectoria
  <video>_lanzamientos.csv  un registro por lanzamiento (zona, al arco / fuera)
  <video>_mapa_arco.png     mapa del arco con todos los lanzamientos
  <video>_arco.json         esquinas del arco (calibración, se reutiliza)

Idea: la cámara es fija, así que las 4 esquinas del arco se marcan una vez.
Con ellas se calcula una homografía imagen -> plano del arco (300 x 200 cm).
El instante en que el balón llega al plano del arco se estima como el punto
de la trayectoria donde su diámetro aparente es más parecido al esperado
para un balón de ~19 cm a la distancia del arco.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ARCO_W, ARCO_H = 300.0, 200.0   # cm, medidas interiores del arco
BALON_D = 19.0                   # cm, balón talla 3 aprox.
COCO_BALL = 32                   # clase 'sports ball' en COCO
FILAS = ["alta", "media", "baja"]
COLS = ["izq", "centro", "der"]
DST = np.float32([[0, 0], [ARCO_W, 0], [ARCO_W, ARCO_H], [0, ARCO_H]])


# ---------------------------------------------------------------- calibración
def calibrar(frame, ruta, forzar):
    if os.path.exists(ruta) and not forzar:
        return np.float32(json.load(open(ruta)))
    pts = []
    nombres = ["superior izquierda", "superior derecha", "inferior derecha", "inferior izquierda"]

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])

    cv2.namedWindow("calibrar", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrar", cb)
    while len(pts) < 4:
        vis = frame.copy()
        for p in pts:
            cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(vis, f"Click esquina {nombres[len(pts)]} (borde interior) | q = salir",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("calibrar", vis)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            sys.exit(0)
    cv2.destroyWindow("calibrar")
    json.dump(pts, open(ruta, "w"))
    return np.float32(pts)


# ---------------------------------------------------------------- detectores
class DetYolo:
    def __init__(self, modelo, conf):
        from ultralytics import YOLO
        self.m, self.conf = YOLO(modelo), conf

    def __call__(self, f):
        r = self.m.predict(f, classes=[COCO_BALL], conf=self.conf, imgsz=1280, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1), float(c)))
        return out  # (x, y, diámetro_px, puntaje)


class DetFondo:
    """Cámara fija: objetos en movimiento = primer plano. Filtra por tamaño y forma."""

    def __init__(self, d_esp):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.d = d_esp

    def __call__(self, f):
        m = self.bg.apply(f)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k)
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        out = []
        for i in range(1, n):
            _, _, w, h, a = st[i]
            d = max(w, h)
            if not (0.5 * self.d <= d <= 4.0 * self.d):
                continue                          # muy chico o muy grande (jugador)
            redondez = min(w, h) / d              # el desenfoque alarga el balón
            relleno = a / float(w * h)
            if redondez < 0.35 or relleno < 0.4:
                continue
            out.append((cen[i][0], cen[i][1], d, redondez * relleno))
        return out


# ---------------------------------------------------------------- seguimiento
class Seguidor:
    """Un balón a la vez: vecino más cercano a la posición predicha (vel. constante)."""

    def __init__(self, gate, max_perdidos, max_quieto):
        self.gate, self.maxp, self.maxq = gate, max_perdidos, max_quieto
        self.pts, self.perdidos = [], 0   # pts: (frame, x, y, d)

    def _cerrar(self):
        fin, self.pts, self.perdidos = self.pts, [], 0
        return fin

    def actualizar(self, n, dets):
        """Devuelve la pista terminada si se cerró en este frame; si no, None."""
        if not self.pts:
            if dets:
                x, y, d, _ = max(dets, key=lambda t: t[3])
                self.pts.append((n, x, y, d))
            return None
        _, x, y, _ = self.pts[-1]
        if len(self.pts) >= 2:
            _, px, py, _ = self.pts[-2]
            x, y = x + (x - px) * (1 + self.perdidos), y + (y - py) * (1 + self.perdidos)
        gate = self.gate * (1 + 0.5 * self.perdidos)
        cand = [(np.hypot(dx - x, dy - y), dx, dy, d) for dx, dy, d, _ in dets]
        cand = [c for c in cand if c[0] < gate]
        if cand:
            _, dx, dy, d = min(cand)
            self.pts.append((n, dx, dy, d))
            self.perdidos = 0
            # balón detenido (en el suelo, en la mano): se descarta la pista
            if len(self.pts) > self.maxq:
                ult = np.array([[p[1], p[2]] for p in self.pts[-self.maxq:]])
                if np.ptp(ult[:, 0]) + np.ptp(ult[:, 1]) < 2 * d:
                    return self._cerrar()
            return None
        self.perdidos += 1
        return self._cerrar() if self.perdidos > self.maxp else None


# ---------------------------------------------------------------- análisis
def a_cm(H, x, y):
    return cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]


def analizar(pista, H, d_esp, fps, k):
    if len(pista) < 5:
        return None
    xy = np.array([[p[1], p[2]] for p in pista])
    if np.ptp(xy[:, 0]) + np.ptp(xy[:, 1]) < 4 * d_esp:
        return None                                   # no hubo lanzamiento
    ds = np.array([p[3] for p in pista], float)
    ds = np.convolve(np.pad(ds, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # suaviza
    i = int(np.argmin(np.abs(ds - d_esp)))            # llegada al plano del arco
    n, x, y, _ = pista[i]
    X, Y = a_cm(H, x, y)
    dentro = 0 <= X <= ARCO_W and 0 <= Y <= ARCO_H
    zona = f"{FILAS[min(int(Y / (ARCO_H / 3)), 2)]}-{COLS[min(int(X / (ARCO_W / 3)), 2)]}" if dentro else "-"
    return dict(n=k, frame=n, t_s=round(n / fps, 2), x_px=round(x), y_px=round(y),
                x_cm=round(float(X), 1), y_cm=round(float(Y), 1),
                resultado="al arco" if dentro else "fuera", zona=zona)


# ---------------------------------------------------------------- dibujo
def dibujar_arco(f, Hinv):
    def p(x, y):
        q = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hinv)[0, 0]
        return int(q[0]), int(q[1])
    esq = [p(*c) for c in DST]
    cv2.polylines(f, [np.int32(esq)], True, (0, 255, 255), 2)
    for i in (1, 2):
        cv2.line(f, p(ARCO_W * i / 3, 0), p(ARCO_W * i / 3, ARCO_H), (0, 200, 200), 1)
        cv2.line(f, p(0, ARCO_H * i / 3), p(ARCO_W, ARCO_H * i / 3), (0, 200, 200), 1)


def mapa(lanz, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, fg, gr = "#1e1e2e", "#cdd6f4", "#6c7086"
    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.add_patch(plt.Rectangle((0, 0), ARCO_W, ARCO_H, fill=False, ec=fg, lw=6))
    for i in (1, 2):
        ax.plot([ARCO_W * i / 3] * 2, [0, ARCO_H], c=gr, ls="--", lw=1)
        ax.plot([0, ARCO_W], [ARCO_H * i / 3] * 2, c=gr, ls="--", lw=1)
    for l in lanz:
        c = "#a6e3a1" if l["resultado"] == "al arco" else "#f38ba8"
        ax.scatter(l["x_cm"], l["y_cm"], s=160, c=c, ec=bg, zorder=3)
        ax.annotate(str(l["n"]), (l["x_cm"], l["y_cm"]), color=bg, ha="center", va="center",
                    fontsize=8, weight="bold", zorder=4)
    xs = [l["x_cm"] for l in lanz] + [0, ARCO_W]
    ys = [l["y_cm"] for l in lanz] + [0, ARCO_H]
    ax.set_xlim(min(xs) - 50, max(xs) + 50)
    ax.set_ylim(max(ys) + 50, min(ys) - 40)   # y hacia abajo: 0 = travesaño
    ax.set_aspect("equal")
    ax.axis("off")
    n_in = sum(l["resultado"] == "al arco" for l in lanz)
    ax.set_title(f"Lanzamientos: {len(lanz)}  ·  al arco: {n_in}  ·  fuera: {len(lanz) - n_in}",
                 color=fg, fontsize=13)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=bg)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--metodo", choices=["yolo", "fondo"], default="yolo")
    ap.add_argument("--modelo", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--recalibrar", action="store_true")
    ap.add_argument("--sin-ventana", action="store_true")
    a = ap.parse_args()

    base = os.path.splitext(a.video)[0]
    cap = cv2.VideoCapture(a.video)
    ok, f0 = cap.read()
    if not ok:
        sys.exit(f"No se pudo leer {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h, w = f0.shape[:2]

    src = calibrar(f0, base + "_arco.json", a.recalibrar)
    H = cv2.getPerspectiveTransform(src, DST)
    Hinv = np.linalg.inv(H)
    ancho_px = (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2
    d_esp = BALON_D / ARCO_W * ancho_px              # diámetro esperado en el plano del arco
    print(f"FPS {fps:.1f} | arco {ancho_px:.0f}px de ancho | balón esperado en el arco ≈ {d_esp:.1f}px")

    det = DetYolo(a.modelo, a.conf) if a.metodo == "yolo" else DetFondo(d_esp)
    seg = Seguidor(gate=0.12 * w, max_perdidos=int(0.15 * fps) + 2, max_quieto=int(fps))
    out = cv2.VideoWriter(base + "_anotado.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    lanz, ultimo, n = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        dets = det(f)
        fin = seg.actualizar(n, dets)
        if fin:
            r = analizar(fin, H, d_esp, fps, len(lanz) + 1)
            if r:
                lanz.append(r)
                ultimo = (r, n)
                print(f"#{r['n']:>2}  t={r['t_s']:>6}s  {r['resultado']:<7}  zona {r['zona']}")

        dibujar_arco(f, Hinv)
        for x, y, d, _ in dets:
            cv2.circle(f, (int(x), int(y)), int(d / 2) + 3, (255, 120, 0), 1)
        if len(seg.pts) > 1:
            cv2.polylines(f, [np.int32([[p[1], p[2]] for p in seg.pts])], False, (0, 0, 255), 2)
        if ultimo and n - ultimo[1] < 2 * fps:
            r = ultimo[0]
            cv2.circle(f, (r["x_px"], r["y_px"]), 10, (0, 255, 0), 3)
            cv2.putText(f, f"#{r['n']} {r['resultado'].upper()}  zona: {r['zona']}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(f, f"t={n / fps:6.2f}s  lanzamientos: {len(lanz)}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(f)
        if not a.sin_ventana:
            cv2.imshow("TEL-328 demo", cv2.resize(f, (min(w, 1280), int(h * min(w, 1280) / w))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1

    if seg.pts:                                        # pista abierta al final del video
        r = analizar(seg.pts, H, d_esp, fps, len(lanz) + 1)
        if r:
            lanz.append(r)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if lanz:
        with open(base + "_lanzamientos.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(lanz[0].keys()))
            wr.writeheader()
            wr.writerows(lanz)
        mapa(lanz, base + "_mapa_arco.png")
    print(f"\n{len(lanz)} lanzamientos -> {base}_anotado.mp4, _lanzamientos.csv, _mapa_arco.png")


if __name__ == "__main__":
    main()
