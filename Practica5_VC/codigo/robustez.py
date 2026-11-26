"""
PRUEBA DE ROBUSTEZ del monitor del conductor

Una sesion de unos 45 s con la camara en una condicion concreta (normal, poca luz, lejos, gafas...).
Calibra 8 s mirando al frente y despues sigue un guion (ojos cerrados, girar, mirar abajo, bostezar).
Al terminar apunta en resultados/robustez.csv que ha detectado el monitor en esa condicion.

Uso:  python codigo/robustez.py normal
      python codigo/robustez.py "poca luz"
"""
import csv
import math
import os
import subprocess
import sys
import time
import cv2
from monitor_conductor import (RAIZ, SEGUNDOS_CALIBRACION, Monitor, calibrar, crear_detector, detectar, medir,
                               dibujar_estado, dibujar_numeros, dibujar_puntos, dibujar_reglas,
                               cerrar_todo, texto)

# (segundo en que empieza, es una accion, instruccion en pantalla)
GUION = [
    (8.0,  False, "NORMAL: no hagas nada raro, parpadea con naturalidad"),
    (15.0, True,  "CIERRA LOS OJOS"),
    (20.0, False, "Mira al frente"),
    (22.0, True,  "GIRA la cabeza a un lado (unos 30-40 grados)"),
    (27.0, False, "Mira al frente"),
    (29.0, True,  "MIRA ABAJO, como si miraras el movil"),
    (34.0, False, "Mira al frente"),
    (36.0, True,  "BOSTEZA (abre mucho la boca)"),
    (41.0, False, "Mira al frente"),
]
FIN = 43.0
GRACIA = 1.5          # segundos extra tras cada accion: la gente reacciona 1-2 s despues del pitido
CAMPOS = ["condicion", "fps", "cara_%", "falsas_alarmas", "ojos", "giro", "abajo", "bostezo"]


def instruccion(t):
    actual = GUION[0]
    for paso in GUION:
        if t >= paso[0]:
            actual = paso
    return actual


def pitido():
    try:
        subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
    except FileNotFoundError:
        pass


def alcanzo(historial, regla, nivel, desde, hasta):
    """True si la regla llego a 'nivel' en algun momento entre 'desde' y 'hasta' (segundos)."""
    return any(n[regla] >= nivel for t, n in historial if desde <= t < hasta)


def evaluar(condicion, historial, fps, pct_cara):
    """Una fila de resultados a partir de lo que paso en la sesion."""
    # falsas alarmas = veces que una regla salta (de 0 a AVISO o mas) mientras no se hace nada raro
    falsas = 0
    previo = {}
    for t, niveles in historial:
        for regla, nivel in niveles.items():
            if 8.0 <= t < 15.0 and nivel >= 1 and previo.get(regla, 0) == 0:
                falsas += 1
            previo[regla] = nivel
    ventana = {paso[2].split()[0]: (paso[0], GUION[i + 1][0]) for i, paso in enumerate(GUION[:-1]) if paso[1]}
    (o1, o2), (g1, g2), (a1, a2), (b1, b2) = ventana["CIERRA"], ventana["GIRA"], ventana["MIRA"], ventana["BOSTEZA"]
    si_no = lambda ok: "si" if ok else "no"
    return {
        "condicion": condicion, "fps": f"{fps:.0f}", "cara_%": f"{pct_cara:.0f}", "falsas_alarmas": falsas,
        "ojos": si_no(alcanzo(historial, "ojos", 2, o1, o2 + GRACIA)),
        "giro": si_no(alcanzo(historial, "giro", 1, g1, g2 + GRACIA)),
        "abajo": si_no(alcanzo(historial, "abajo", 1, a1, a2 + GRACIA)),
        "bostezo": si_no(alcanzo(historial, "bostezo", 1, b1, b2 + GRACIA)),
    }


def guardar(fila, ruta=os.path.join(RAIZ, "resultados", "robustez.csv")):
    """Guarda la fila; si esa condicion ya estaba, se sustituye por la nueva."""
    filas = {}
    if os.path.exists(ruta):
        with open(ruta, newline="") as f:
            filas = {r["condicion"]: r for r in csv.DictReader(f)}
    filas[fila["condicion"]] = fila
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas.values())


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Uso: python codigo/robustez.py "nombre de la condicion"')
    condicion = sys.argv[1]
    detector = crear_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cerrar_todo(cap, detector)
        raise SystemExit("No puedo abrir la camara")

    monitor, muestras, historial = None, [], []
    n_frames, n_con_cara, paso_ant = 0, 0, None
    t0 = time.time()
    ultimo_ts = -1
    resultado = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            t = time.time() - t0
            ts = max(int(t * 1000), ultimo_ts + 1)
            ultimo_ts = ts
            pts = detectar(detector, frame, ts)
            s = medir(pts, w, h)
            n_frames += 1
            n_con_cara += pts is not None
            if pts is not None:
                dibujar_puntos(frame, pts)

            if monitor is None:                               # primeros 8 s: calibrar
                if not math.isnan(s["ear"]):
                    muestras.append(s)
                cv2.rectangle(frame, (0, 0), (w, 44), (150, 100, 40), -1)
                texto(frame, f"CALIBRANDO ({condicion}): mira al frente, sin hablar", (15, 31), 0.7)
                if t >= SEGUNDOS_CALIBRACION:
                    base, aviso = calibrar(muestras, n_frames)
                    if base is None:
                        print(f"NO CALIBRA en la condicion '{condicion}': {aviso}")
                        resultado = {"condicion": condicion, "fps": f"{n_frames / t:.0f}",
                                     "cara_%": f"{100 * n_con_cara / n_frames:.0f}", "falsas_alarmas": "-",
                                     "ojos": "-", "giro": "-", "abajo": "-", "bostezo": "-"}
                        break
                    monitor = Monitor(base)
            else:
                niveles, n = monitor.evaluar(t, s)
                historial.append((t, niveles))
                dibujar_estado(frame, niveles)
                dibujar_numeros(frame, n)
                dibujar_reglas(frame, niveles)
                paso = instruccion(t)
                if paso is not paso_ant:
                    pitido()
                    paso_ant = paso
                texto(frame, paso[2], (15, h - 70), 0.7, (0, 255, 255) if paso[1] else (255, 255, 255))
                if t >= FIN:
                    break

            cv2.imshow("Prueba de robustez", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("Prueba cancelada")
                break
    except KeyboardInterrupt:
        print("Interrumpido con Ctrl+C")
    finally:
        cerrar_todo(cap, detector)

    if resultado is None and historial and historial[-1][0] >= FIN:
        resultado = evaluar(condicion, historial, n_frames / (time.time() - t0), 100 * n_con_cara / n_frames)
    if resultado is not None:
        guardar(resultado)
        print("Resultado:", resultado)


if __name__ == "__main__":
    main()
