#!/usr/bin/env python3
"""
TEL-328 · Detección del balón y del arco en video grabado con cámara en mano.

Pipeline (procesamiento clásico, sin redes neuronales):
  1. Estabilización: puntos ORB en la zona estática (pared/arco) + homografía RANSAC
     de cada cuadro a un cuadro de referencia. Así el arco queda fijo.
  2. Arco: 4 esquinas marcadas una vez en el cuadro de referencia -> homografía al
     plano real del arco (300 x 200 cm) y grilla de 3x3 zonas.
  3. Balón: diferencia de 3 cuadros sobre el video estabilizado, umbral, morfología y
     filtro por tamaño; se descartan manchas pegadas a objetos grandes (jugadores).
  4. Seguimiento: unión de candidatos con modelo de velocidad constante.
     La pista del balón es la más larga con desplazamiento rápido.
  5. Análisis: punto de entrada al arco, zona y resultado (gol si, después de
     entrar, el balón se frena dentro del arco: la red lo detiene).

Uso:
  python pipeline_balon.py video.mp4 --ref 180 --esquinas 37,399 128,399 128,487 37,490 --hasta 330
"""
import argparse
import csv
import os
import shutil
import subprocess

import cv2
import numpy as np

ARCO_W, ARCO_H = 300.0, 200.0
FILAS, COLS = ["alta", "media", "baja"], ["izq", "centro", "der"]
DST = np.float32([[0, 0], [ARCO_W, 0], [ARCO_W, ARCO_H], [0, ARCO_H]])


def leer(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr, fps


def estabilizar(frames, ref, y_max):
    """Homografía de cada cuadro al cuadro de referencia (zona superior = estática)."""
    h, w = frames[0].shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[:y_max] = 255
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    gris = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    kr, dr = orb.detectAndCompute(gris(frames[ref]), mask)
    Hs = []
    for f in frames:
        k, d = orb.detectAndCompute(gris(f), mask)
        H = None
        if d is not None and len(k) > 20:
            m = [a for a, b in bf.knnMatch(d, dr, k=2) if a.distance < 0.75 * b.distance]
            if len(m) >= 15:
                src = np.float32([k[a.queryIdx].pt for a in m])
                dst = np.float32([kr[a.trainIdx].pt for a in m])
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        Hs.append(H if H is not None else (Hs[-1] if Hs else np.eye(3)))
    return Hs


def candidatos(G, t, roi, th=22):
    """Diferencia de 3 cuadros: objetos que se movieron en t-2 -> t y en t -> t+2."""
    x0, x1, y0, y1 = roi
    a = cv2.absdiff(G[t], G[t - 2])
    b = cv2.absdiff(G[t + 2], G[t])
    m = (((a > th) & (b > th)).astype(np.uint8)) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    grande = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 120]).astype(np.uint8)
    grande = cv2.dilate(grande, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    out = []
    for i in range(1, n):
        _, _, w, h, A = st[i]
        cx, cy = cen[i]
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        if A < 6 or A > 90 or max(w, h) > 14:      # tamaño de balón a esta distancia
            continue
        if grande[int(cy), int(cx)]:               # pegado a un jugador
            continue
        out.append((float(cx), float(cy)))
    return out, m


def seguir(C, t0, t1):
    activas, cerradas = [], []
    for t in range(t0, t1):
        usados = set()
        for tr in activas:
            tl, xl, yl = tr[-1]
            vx = vy = 0.0
            if len(tr) >= 2:
                tp, xp, yp = tr[-2]
                vx, vy = (xl - xp) / (tl - tp), (yl - yp) / (tl - tp)
            dt = t - tl
            px, py = xl + vx * dt, yl + vy * dt
            gate = 6 + 4 * dt if len(tr) >= 2 else 16
            mejor = None
            for j, (x, y) in enumerate(C[t]):
                d = np.hypot(x - px, y - py)
                if j not in usados and d < gate and (mejor is None or d < mejor[0]):
                    mejor = (d, j)
            if mejor:
                usados.add(mejor[1])
                tr.append((t, *C[t][mejor[1]]))
        sig = []
        for tr in activas:
            (cerradas if t - tr[-1][0] > 4 else sig).append(tr)
        activas = sig + [[(t, x, y)] for j, (x, y) in enumerate(C[t]) if j not in usados]
    return cerradas + activas


def vel_max(tr, k=3):
    p = np.array(tr)
    if len(p) <= k:
        return 0
    return max(np.hypot(p[i + k, 1] - p[i, 1], p[i + k, 2] - p[i, 2]) / (p[i + k, 0] - p[i, 0])
               for i in range(len(p) - k))


def vuelo(tr, v_min=7.0, cos_min=0.8):
    """Racha más larga de pasos rápidos y en la misma dirección (balón en vuelo)."""
    p = np.array(tr)
    if len(p) < 3:
        return 0
    d = np.diff(p[:, 1:], axis=0) / np.diff(p[:, 0])[:, None]
    v = np.hypot(d[:, 0], d[:, 1])
    mejor = run = 0
    for i in range(len(d)):
        ok = v[i] > v_min
        if ok and run > 0:
            c = d[i] @ d[i - 1] / (v[i] * v[i - 1] + 1e-9)
            ok = c > cos_min
            run = run + 1 if ok else (1 if v[i] > v_min else 0)
        else:
            run = 1 if ok else 0
        mejor = max(mejor, run)
    return mejor


def dentro(poly, x, y):
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ref", type=int, required=True, help="cuadro de referencia")
    ap.add_argument("--esquinas", nargs=4, required=True,
                    help="x,y de esquinas interiores: sup-izq sup-der inf-der inf-izq (cuadro ref)")
    ap.add_argument("--hasta", type=int, default=None, help="último cuadro a procesar")
    ap.add_argument("--salida", default="salida")
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    frames, fps = leer(a.video)
    N = min(a.hasta or len(frames), len(frames) - 3)
    h, w = frames[0].shape[:2]
    esq = np.float32([list(map(float, e.split(","))) for e in a.esquinas])
    Ha = cv2.getPerspectiveTransform(esq, DST)
    xs, ys = esq[:, 0], esq[:, 1]
    gw, gh = xs.max() - xs.min(), ys.max() - ys.min()
    roi = (max(0, xs.min() - gw * 0.4), min(w, xs.max() + gw * 1.5),
           max(0, ys.min() - gh * 0.8), min(h, ys.max() + gh * 0.15))

    print("1/4 estabilizando...")
    Hs = estabilizar(frames[:N + 3], a.ref, int(ys.max() - gh * 0.1))
    est = [cv2.warpPerspective(frames[i], Hs[i], (w, h)) for i in range(N + 3)]
    G = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0) for f in est]

    print("2/4 detectando candidatos...")
    C, M = {}, {}
    for t in range(2, N):
        C[t], M[t] = candidatos(G, t, roi)

    print("3/4 siguiendo...")
    pistas = [p for p in seguir(C, 2, N) if vuelo(p) >= 4]
    if not pistas:
        print("No se encontró una pista de balón.")
        return
    balon = max(pistas, key=lambda p: (vuelo(p), len(p)))

    print("4/4 analizando...")
    entrada, k_in, frenado = None, 0, False
    for i in range(1, len(balon)):
        (t0, x0, y0), (t1, x1, y1) = balon[i - 1], balon[i]
        if not dentro(esq, x0, y0) and dentro(esq, x1, y1):
            # punto de cruce del borde: interpolación lineal
            for s in np.linspace(0, 1, 50):
                if dentro(esq, x0 + (x1 - x0) * s, y0 + (y1 - y0) * s):
                    entrada = (t0 + (t1 - t0) * s, x0 + (x1 - x0) * s, y0 + (y1 - y0) * s)
                    break
            # GOL si la red frena el balón: dentro del arco y con velocidad mucho menor
            pa = np.array(balon[max(0, i - 4):i + 1])
            pd_ = np.array(balon[i:i + 8])
            v_antes = np.median(np.hypot(np.diff(pa[:, 1]), np.diff(pa[:, 2])) / np.diff(pa[:, 0]))
            v_desp = np.median(np.hypot(np.diff(pd_[:, 1]), np.diff(pd_[:, 2])) / np.diff(pd_[:, 0]))
            k_in = sum(dentro(esq, x, y) for _, x, y in balon[i:i + 8])
            frenado = v_desp < 0.4 * v_antes
            break
    res = {"resultado": "sin llegada al arco", "zona": "-"}
    if entrada:
        X, Y = cv2.perspectiveTransform(np.float32([[entrada[1:]]]), Ha)[0, 0]
        X, Y = float(np.clip(X, 0, ARCO_W - 1e-3)), float(np.clip(Y, 0, ARCO_H - 1e-3))
        zona = f"{FILAS[int(Y // (ARCO_H / 3))]}-{COLS[int(X // (ARCO_W / 3))]}"
        res = {"resultado": "GOL" if (frenado and k_in >= 4) else "al arco, sin gol",
               "zona": zona, "t_entrada_s": round(float(entrada[0]) / fps, 2),
               "x_cm": round(X), "y_cm": round(Y)}
    vel_px = vel_max(balon)
    print(f"Pista balón: cuadros {balon[0][0]}-{balon[-1][0]} ({len(balon)} detecciones), "
          f"vel. máx {vel_px:.1f} px/cuadro")
    print("Resultado:", res)

    with open(f"{a.salida}/trayectoria_balon.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "t_s", "x_px_ref", "y_px_ref", "dentro_arco"])
        for t, x, y in balon:
            wr.writerow([t, round(t / fps, 3), round(x, 1), round(y, 1), int(dentro(esq, x, y))])

    # ---------------------------------------------------------------- video
    Hai = np.linalg.inv(Ha)
    grid = []
    for i in range(4):
        grid.append(((ARCO_W * i / 3, 0), (ARCO_W * i / 3, ARCO_H)))
        grid.append(((0, ARCO_H * i / 3), (ARCO_W, ARCO_H * i / 3)))
    pos = {t: (x, y) for t, x, y in balon}
    t_in = entrada[0] if entrada else 1e9

    def proyectar(pts, Hm):
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hm).reshape(-1, 2)

    x0, x1, y0, y1 = map(int, roi)
    y0c, y1c = max(0, y0 - 20), min(h, y1 + 20)
    Z = 3
    zw, zh = (x1 - x0) * Z, (y1c - y0c) * Z
    tmp1, tmp2 = f"{a.salida}/_full.mp4", f"{a.salida}/_zoom.mp4"
    ini, fin = max(0, balon[0][0] - 45), min(N, balon[-1][0] + 20)
    vw1 = cv2.VideoWriter(tmp1, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
    vw2 = cv2.VideoWriter(tmp2, cv2.VideoWriter_fourcc(*"mp4v"), fps / 4, (2 * zw, zh))  # cámara lenta x4
    grid_ref = [proyectar([p, q], Hai) for p, q in grid]

    def rotulo(img, sx):
        if t >= t_in:
            cv2.rectangle(img, (0, 0), (img.shape[1], int(70 * sx)), (0, 0, 0), -1)
            cv2.putText(img, res["resultado"], (10, int(30 * sx)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sx, (0, 255, 0), 2)
            cv2.putText(img, f"zona: {res['zona']}", (10, int(60 * sx)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sx, (0, 255, 0), 2)

    for t in range(N):
        hist = [(x, y) for tt, x, y in balon if tt <= t]
        # --- video completo: cuadro original, arco seguido con la homografía inversa
        f = frames[t].copy()
        Hinv = np.linalg.inv(Hs[t])
        for pq in grid_ref:
            pq = proyectar(pq, Hinv).astype(int)
            cv2.line(f, tuple(pq[0]), tuple(pq[1]), (0, 255, 255), 1)
        cv2.polylines(f, [proyectar(esq, Hinv).astype(np.int32)], True, (0, 255, 255), 2)
        if len(hist) > 1:
            cv2.polylines(f, [proyectar(hist, Hinv).astype(np.int32)], False, (0, 0, 255), 2)
        if t in pos:
            cv2.circle(f, tuple(proyectar([pos[t]], Hinv)[0].astype(int)), 8, (0, 255, 0), 2)
        rotulo(f, 1)
        cv2.putText(f, f"t={t / fps:5.2f}s", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        m = M.get(t, np.zeros((h, w), np.uint8))
        mv = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mv, (x0, y0), (x1, y1), (255, 120, 0), 1)
        cv2.polylines(mv, [esq.astype(np.int32)], True, (0, 255, 255), 1)
        if t in pos:
            cv2.circle(mv, tuple(map(int, pos[t])), 8, (0, 255, 0), 2)
        cv2.putText(mv, "movimiento (estabilizado)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw1.write(np.hstack([f, mv]))
        # --- zoom al arco sobre el video estabilizado
        e = cv2.resize(est[t][y0c:y1c, x0:x1], (zw, zh), interpolation=cv2.INTER_CUBIC)
        mz = cv2.resize(mv[y0c:y1c, x0:x1].copy() * 0 + cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)[y0c:y1c, x0:x1],
                        (zw, zh), interpolation=cv2.INTER_NEAREST)
        off = np.float32([x0, y0c])
        for pq in grid_ref:
            pq = ((pq - off) * Z).astype(int)
            cv2.line(e, tuple(pq[0]), tuple(pq[1]), (0, 255, 255), 1)
        for img in (e, mz):
            cv2.polylines(img, [((esq - off) * Z).astype(np.int32)], True, (0, 255, 255), 2)
            if len(hist) > 1:
                cv2.polylines(img, [((np.float32(hist) - off) * Z).astype(np.int32)], False, (0, 0, 255), 2)
            if t in pos:
                cv2.circle(img, tuple(((np.float32(pos[t]) - off) * Z).astype(int)), 14, (0, 255, 0), 2)
        rotulo(e, 1.3)
        cv2.putText(e, f"t={t / fps:5.2f}s", (10, zh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(mz, "diferencia de 3 cuadros", (10, zh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if ini <= t < fin:
            vw2.write(np.hstack([e, mz]))
    vw1.release()
    vw2.release()
    # H.264 si hay ffmpeg (se reproduce en cualquier lado); si no, queda en mp4v
    ffmpeg = shutil.which("ffmpeg")
    for src, nombre in [(tmp1, "demo_balon.mp4"), (tmp2, "demo_zoom_lento.mp4")]:
        dst = os.path.join(a.salida, nombre)
        if ffmpeg:
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", src, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", dst], check=True)
            os.remove(src)
        else:
            os.replace(src, dst)

    # ---------------------------------------------------------------- figura
    ref = est[a.ref].copy()
    cv2.polylines(ref, [esq.astype(np.int32)], True, (0, 255, 255), 2)
    for p, q in grid:
        pq = proyectar([p, q], Hai).astype(int)
        cv2.line(ref, tuple(pq[0]), tuple(pq[1]), (0, 255, 255), 1)
    pts = np.int32([(x, y) for _, x, y in balon])
    cv2.polylines(ref, [pts], False, (0, 0, 255), 2)
    for x, y in pts[::2]:
        cv2.circle(ref, (int(x), int(y)), 2, (0, 0, 255), -1)
    if entrada:
        cv2.circle(ref, (int(entrada[1]), int(entrada[2])), 7, (0, 255, 0), 2)
    x0, x1, y0, y1 = map(int, roi)
    crop = ref[max(0, y0 - 20):y1 + 20, max(0, x0):x1]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{a.salida}/trayectoria_balon.png", crop)
    with open(f"{a.salida}/resultado.txt", "w") as fh:
        fh.write(f"{res}\npista: cuadros {balon[0][0]}-{balon[-1][0]}, {len(balon)} detecciones\n")
    print("Listo ->", a.salida)


if __name__ == "__main__":
    main()