#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconocimiento de figuras geometricas como PROBLEMA DE BUSQUEDA.
  Estado inicial : imagen cruda (sin rasgos evaluados)
  Estados        : interpretacion parcial = conjunto de rasgos ya detectados + candidatos vivos
  Acciones       : detectar lados, circularidad, solidez, simetrias, angulos, aspecto
  Objetivo       : los rasgos coinciden con UNA de las figuras (circulo, cuadrado,
                   triangulo, trapecio, estrella) -> si no coincide ninguna: "nulo"
  Coste          : numero de comparaciones + rasgos no coincidentes

Uso:
  python figuras.py imagen.png                  # A* (heuristica)
  python figuras.py imagen.png --metodo bfs     # busqueda ciega
  python figuras.py --demo estrella             # genera una figura de prueba
  python figuras.py --calibrar                  # algoritmo genetico: ajusta pesos
Requiere: pip install opencv-python numpy matplotlib
"""
import argparse, heapq, itertools, json, math, os, sys
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt

FIGURAS = ["circulo", "cuadrado", "triangulo", "trapecio", "estrella"]
RASGOS = ["lados", "circularidad", "solidez", "simetrias", "desv_angulo", "aspecto"]

# Prototipos: rango (min, max) esperado de cada rasgo por figura
PROTO = {
    "circulo":  dict(lados=(8, 99), circularidad=(.85, 1.1), solidez=(.95, 1), simetrias=(6, 99), desv_angulo=(0, 180), aspecto=(.88, 1.12)),
    "cuadrado": dict(lados=(4, 4), circularidad=(.70, .85), solidez=(.95, 1), simetrias=(3, 6), desv_angulo=(0, 8), aspecto=(.85, 1.12)),
    "triangulo": dict(lados=(3, 3), circularidad=(.50, .68), solidez=(.95, 1), simetrias=(1, 3), desv_angulo=(0, 45), aspecto=(.3, 1.2)),
    "trapecio": dict(lados=(4, 4), circularidad=(.60, .80), solidez=(.95, 1), simetrias=(0, 2), desv_angulo=(8, 45), aspecto=(.45, 1.0)),
    "estrella": dict(lados=(8, 12), circularidad=(0, .50), solidez=(.40, .75), simetrias=(3, 6), desv_angulo=(0, 180), aspecto=(.8, 1.15)),
}
TOL = dict(lados=1.8, circularidad=.15, solidez=.1, simetrias=2, desv_angulo=5, aspecto=.12)
W0 = {f: 1.0 for f in RASGOS}
UMBRAL = 0.62
PESOS_JSON = "pesos.json"


# ------------------------------------------------------------------ RASGOS (OpenCV)
def segmentar(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borde = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    if borde.mean() > 127:           # fondo blanco -> figura oscura: invertir
        b = cv2.bitwise_not(b)
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def simetrias(c):
    """Cuenta ejes de simetria: refleja la mascara respecto a ejes cada 2 grados."""
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    x, y, w, h = cv2.boundingRect(c)
    s = 160.0 / max(w, h)
    pts = ((c.reshape(-1, 2) - [cx, cy]) * s + 100).astype(np.int32)
    m = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    hits = []
    for t in range(0, 180, 2):
        a = 2 * math.radians(t)
        R = np.array([[math.cos(a), math.sin(a)], [math.sin(a), -math.cos(a)]])
        off = np.array([100, 100]) - R @ np.array([100, 100])
        mr = cv2.warpAffine(m, np.hstack([R, off.reshape(2, 1)]), (200, 200), flags=cv2.INTER_NEAREST)
        inter, uni = np.logical_and(m > 0, mr > 0).sum(), np.logical_or(m > 0, mr > 0).sum()
        hits.append(inter / uni > 0.92)
    hits = np.array(hits)
    if hits.all():
        return 99
    return int((hits & ~np.roll(hits, 1)).sum())


def extraer(img):
    """Devuelve (rasgos, info_visual). rasgos=None si no hay figura."""
    b = segmentar(img)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, dict(bin=b)
    c = max(cs, key=cv2.contourArea)
    A = cv2.contourArea(c)
    if A < 0.005 * b.size:
        return None, dict(bin=b)
    P = cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, 0.025 * P, True).reshape(-1, 2)
    hull = cv2.convexHull(c)
    n = len(ap)
    angs = []
    for i in range(n):
        v1, v2 = ap[i - 1] - ap[i], ap[(i + 1) % n] - ap[i]
        cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angs.append(math.degrees(math.acos(np.clip(cosv, -1, 1))))
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    F = dict(
        lados=n,
        circularidad=float(4 * math.pi * A / (P * P)),
        solidez=float(A / (cv2.contourArea(hull) + 1e-9)),
        simetrias=simetrias(c),
        desv_angulo=float(np.mean(np.abs(np.array(angs) - 90))),
        aspecto=float(min(rw, rh) / (max(rw, rh) + 1e-9)),
    )
    F["_area"], F["_perimetro"], F["_angulos"] = float(A), float(P), [round(a) for a in angs]
    return F, dict(bin=b, c=c, ap=ap, hull=hull)


# ------------------------------------------------------------------ BUSQUEDA
def puntaje(f, v, fig):
    lo, hi = PROTO[fig][f]
    d = 0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi))
    return max(0.0, 1 - d / TOL[f])


def busqueda(F, metodo="astar"):
    """Busca en el espacio de interpretaciones parciales (conjuntos de rasgos evaluados).
    bfs   : ciega, amplitud, evalua todas las figuras con todos los rasgos.
    astar : heuristica, poda candidatos con >=2 desajustes; h = candidatos vivos - 1."""
    S = {f: {g: puntaje(f, F[f], g) for g in FIGURAS} for f in RASGOS}
    mism = lambda est, g: sum(S[f][g] < .5 for f in est)
    vivos = lambda est: [g for g in FIGURAS if mism(est, g) < 2]

    def coste(est, f):
        v = vivos(est) if metodo == "astar" else FIGURAS
        return len(v) + sum(S[f][g] < .5 for g in v)      # comparaciones + desajustes

    ini, cnt, exp = frozenset(), itertools.count(), 0
    if metodo == "bfs":
        cola, vistos = deque([(ini, 0, ())]), {ini}
        while cola:
            est, g, ruta = cola.popleft(); exp += 1
            if len(est) == len(RASGOS):
                return dict(ruta=ruta, coste=g, expandidos=exp, vivos=vivos(est))
            for f in RASGOS:
                if f not in est and (est | {f}) not in vistos:
                    vistos.add(est | {f}); cola.append((est | {f}, g + coste(est, f), ruta + (f,)))
    pq, mejor = [(0, 0, 0, ini, ())], {ini: 0}
    while pq:
        _, _, g, est, ruta = heapq.heappop(pq); exp += 1
        v = vivos(est)
        if len(v) <= 1 or len(est) == len(RASGOS):       # prueba de objetivo
            return dict(ruta=ruta, coste=g, expandidos=exp, vivos=v)
        for f in RASGOS:
            if f in est:
                continue
            ne, ng = est | {f}, g + coste(est, f)
            if ng < mejor.get(ne, 1e9):
                mejor[ne] = ng
                h = max(0, len(vivos(ne)) - 1)
                heapq.heappush(pq, (ng + h, next(cnt), ng, ne, ruta + (f,)))


def confianzas(F, W):
    """Similitud ponderada; cada rasgo no coincidente penaliza (el numero de lados, mas)."""
    out = {}
    for g in FIGURAS:
        sc = {f: puntaje(f, F[f], g) for f in RASGOS}
        c = sum(W[f] * sc[f] for f in RASGOS) / sum(W.values())
        for f in RASGOS:
            if sc[f] < .5:
                c *= 0.6 if f == "lados" else 0.75
        out[g] = c
    return out


def clasificar(F, metodo="astar", W=W0):
    if F is None:
        return "nulo", 0.0, {g: 0.0 for g in FIGURAS}, None
    r = busqueda(F, metodo)
    conf = confianzas(F, W)
    cand = r["vivos"] or FIGURAS
    mejor = max(cand, key=conf.get)
    if conf[mejor] < UMBRAL:
        return "nulo", 1 - conf[mejor], conf, r
    return mejor, conf[mejor], conf, r


# ------------------------------------------------------------------ ALGORITMO GENETICO
def sintetica(fig, rng, n=300):
    img = np.full((n, n, 3), 255, np.uint8)
    cx, cy = n // 2 + rng.integers(-20, 20), n // 2 + rng.integers(-20, 20)
    r, rot = rng.integers(60, 105), rng.uniform(0, 2 * math.pi)
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    def poli(pts):
        p = (np.array(pts, float) * r) @ R.T + [cx, cy]
        cv2.fillPoly(img, [p.astype(np.int32)], (0, 0, 0))
    ang = lambda k, i: 2 * math.pi * i / k
    if fig == "circulo":
        cv2.circle(img, (cx, cy), r, (0, 0, 0), -1)
    elif fig == "cuadrado":
        poli([(-.8, -.8), (.8, -.8), (.8, .8), (-.8, .8)])
    elif fig == "triangulo":
        poli([(math.cos(ang(3, i)), math.sin(ang(3, i))) for i in range(3)])
    elif fig == "trapecio":
        poli([(-1, .6), (1, .6), (.5, -.6), (-.5, -.6)])
    elif fig == "estrella":
        poli([((1 if i % 2 == 0 else .4) * math.cos(ang(10, i)), (1 if i % 2 == 0 else .4) * math.sin(ang(10, i))) for i in range(10)])
    elif fig == "nulo":
        poli([(-1.1, -.45), (1.1, -.45), (1.1, .45), (-1.1, .45)])
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def calibrar(n_por_clase=25, pob=24, gens=25, seed=1):
    """Algoritmo genetico: cromosoma = pesos de los 6 rasgos; fitness = exactitud."""
    rng = np.random.default_rng(seed)
    datos = []
    for fig in FIGURAS + ["nulo"]:
        for _ in range(n_por_clase):
            F, _ = extraer(sintetica(fig, rng))
            datos.append((F, fig))
    def fitness(w):
        W = dict(zip(RASGOS, w))
        return np.mean([clasificar(F, "astar", W)[0] == y for F, y in datos])
    P = rng.uniform(.1, 2, (pob, len(RASGOS)))
    for g in range(gens):
        fit = np.array([fitness(w) for w in P])
        print(f"gen {g:2d}  mejor={fit.max():.3f}  media={fit.mean():.3f}")
        nuevos = [P[fit.argmax()].copy()]                          # elitismo
        while len(nuevos) < pob:
            a, b = [P[max(rng.integers(0, pob, 3), key=lambda i: fit[i])] for _ in range(2)]  # torneo
            hijo = np.where(rng.random(len(RASGOS)) < .5, a, b)    # cruce uniforme
            hijo = np.clip(hijo + rng.normal(0, .15, len(RASGOS)) * (rng.random(len(RASGOS)) < .3), .05, 3)  # mutacion
            nuevos.append(hijo)
        P = np.array(nuevos)
    mejor = P[np.array([fitness(w) for w in P]).argmax()]
    W = {f: round(float(v), 3) for f, v in zip(RASGOS, mejor)}
    json.dump(W, open(PESOS_JSON, "w"))
    print("Pesos optimos guardados en", PESOS_JSON, W)
    return W


# ------------------------------------------------------------------ VISUALIZACION
def mostrar(img, F, vis, fig, conf, confs, r, metodo, salida):
    ft = plt.figure(figsize=(13, 8))
    ax = ft.add_subplot(2, 2, 1); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_title("Imagen original"); ax.axis("off")
    ax = ft.add_subplot(2, 2, 2)
    canvas = cv2.cvtColor(vis["bin"], cv2.COLOR_GRAY2BGR)
    if F is not None:
        cv2.drawContours(canvas, [vis["c"]], -1, (0, 200, 0), 2)
        cv2.polylines(canvas, [vis["hull"]], True, (255, 120, 0), 1)
        for p in vis["ap"]:
            cv2.circle(canvas, tuple(int(v) for v in p), 6, (0, 0, 255), -1)
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title("Rasgos: contorno (verde), casco convexo (azul), vertices (rojo)")
    ax = ft.add_subplot(2, 2, 3)
    ax.barh(FIGURAS, [confs[g] * 100 for g in FIGURAS], color=["#1D9E75" if g == fig else "#B4B2A9" for g in FIGURAS])
    ax.set_xlim(0, 100); ax.set_xlabel("Similitud con prototipo (%)"); ax.set_title("Puntaje por figura")
    ax = ft.add_subplot(2, 2, 4); ax.axis("off")
    txt = f"FIGURA IDENTIFICADA: {fig.upper()}\nCONFIANZA: {conf * 100:.1f}%\n\n"
    if F is not None:
        txt += (f"lados/vertices : {F['lados']}\nangulos (deg)  : {F['_angulos']}\n"
                f"desv. a 90 deg : {F['desv_angulo']:.1f}\nsimetrias      : {F['simetrias']}\n"
                f"area           : {F['_area']:.0f} px2\nperimetro      : {F['_perimetro']:.0f} px\n"
                f"circularidad   : {F['circularidad']:.3f}\nsolidez        : {F['solidez']:.3f}\n"
                f"aspecto        : {F['aspecto']:.2f}\n\n")
        txt += (f"Busqueda: {metodo}\nruta de rasgos : {' > '.join(r['ruta'])}\n"
                f"coste total    : {r['coste']}\nnodos expandidos: {r['expandidos']}")
    else:
        txt += "No se detecto ninguna figura en la imagen."
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=10)
    ft.tight_layout(); ft.savefig(salida, dpi=110); print("Resultado guardado en", salida)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen", nargs="?")
    ap.add_argument("--metodo", choices=["astar", "bfs"], default="astar")
    ap.add_argument("--demo", choices=FIGURAS + ["nulo"])
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--salida", default="resultado.png")
    a = ap.parse_args()
    if a.calibrar:
        calibrar(); return
    if a.demo:
        img = sintetica(a.demo, np.random.default_rng(7))
    elif a.imagen:
        img = cv2.imread(a.imagen)
        if img is None:
            sys.exit("No se pudo leer la imagen: " + a.imagen)
    else:
        ap.print_help(); return
    W = json.load(open(PESOS_JSON)) if os.path.exists(PESOS_JSON) else W0
    F, vis = extraer(img)
    fig, conf, confs, r = clasificar(F, a.metodo, W)
    print(f"Figura: {fig}  confianza: {conf * 100:.1f}%")
    mostrar(img, F, vis, fig, conf, confs, r, a.metodo, a.salida)


if __name__ == "__main__":
    main()
