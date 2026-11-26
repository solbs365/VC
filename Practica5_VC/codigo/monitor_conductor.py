"""
PROTOTIPO B: Monitor de atencion del conductor (en directo)

Vigila la cara con MediaPipe y avisa si el conductor parece distraido o con sueno:
ojos cerrados, cabeza hacia abajo, mirada a otro lado, bostezo o cara fuera de camara.
Al arrancar se calibra con la cara de cada persona (8 s mirando al frente), asi que
los umbrales se miden respecto a "tu normal" y no a valores fijos.

Uso:    python codigo/monitor_conductor.py
        python codigo/monitor_conductor.py --grabar resultados/demo_monitor.mp4      (guarda tambien el video)
Teclas: c = calibrar otra vez     m = sonido si/no     ESC o q = salir
"""
import argparse
import csv
import math
import os
import subprocess
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # carpeta Practica5_VC

# ---------- puntos de la malla de la cara (los mismos del cuaderno) ----------
OJO_DER = [33, 160, 158, 133, 153, 144]
OJO_IZQ = [362, 385, 387, 263, 373, 380]
BOCA = [78, 81, 13, 311, 308, 402, 14, 178]
PNP = [1, 152, 33, 263, 133, 362, 61, 291, 10]   # nariz, barbilla, esquinas de ojos, comisuras, frente
CARA_3D = np.array([
    [0.000, -1.127, 7.476],   # 1
    [0.000, -9.403, 4.264],   # 152
    [-4.446, 2.664, 3.173],   # 33
    [4.446, 2.664, 3.173],    # 263
    [-1.856, 2.585, 3.758],   # 133
    [1.856, 2.585, 3.758],    # 362
    [-2.456, -4.343, 4.284],  # 61
    [2.456, -4.343, 4.284],   # 291
    [0.000, 8.262, 4.482],    # 10
]) * np.array([1, -1, -1])                        # al sistema de OpenCV: y hacia abajo, z hacia el fondo

# ---------- umbrales y reglas (los mismos del cuaderno) ----------
UMBRAL_OJOS = 0.5            # EAR_rel por debajo de esto = ojos cerrados
UMBRAL_ABAJO = 15            # grados mirando abajo respecto a tu postura normal
UMBRAL_GIRO = 20             # grados de giro respecto a tu postura normal
UMBRAL_BOCA_ABIERTA = 0.15   # MAR por encima de esto = boca abierta
UMBRAL_BOCA_CERRADA = 0.10   # la boca se da por cerrada por debajo de esto (histeresis)
MAX_YAW = 40                 # con mas giro, EAR y MAR dejan de ser fiables
SEGUNDOS_CALIBRACION = 8.0

# nombre: (segundos para AVISO, segundos para ALERTA, hueco de datos que se tolera en segundos)
REGLAS = {
    "ojos":     (0.5, 1.0, 0.0),
    "abajo":    (1.0, 2.0, 0.3),
    "giro":     (1.0, 2.0, 0.3),
    "bostezo":  (1.5, math.inf, 0.0),
    "sin cara": (1.0, 2.0, 0.0),
}
TEXTO_MOTIVO = {
    "ojos": "ojos cerrados",
    "abajo": "mirando abajo",
    "giro": "mirando a otro lado",
    "bostezo": "bostezo",
    "sin cara": "no veo la cara",
}


# ---------- las senales de un fotograma (identicas a las del cuaderno) ----------
def ear_ojo(pts, idx):
    p1, p2, p3, p4, p5, p6 = pts[idx]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal)


def ear(pts):
    return (ear_ojo(pts, OJO_DER) + ear_ojo(pts, OJO_IZQ)) / 2


def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))


def mar(pts):
    vertical = dist(pts, 81, 178) + dist(pts, 13, 14) + dist(pts, 311, 402)
    horizontal = dist(pts, 78, 308)
    return vertical / (3.0 * horizontal)


def pose_cabeza(pts, w, h):
    """(yaw, pitch, roll) en grados; None si la solucion no vale."""
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(CARA_3D, pts[PNP].astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:                  # la cara no puede estar detras de la camara
        return None
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def medir(pts, w, h):
    """Las 5 senales de UN fotograma. Si no hay cara (pts es None), todo nan."""
    s = {k: np.nan for k in ("ear", "mar", "yaw", "pitch", "roll")}
    if pts is None:
        return s
    s["ear"] = ear(pts)
    s["mar"] = mar(pts)
    pose = pose_cabeza(pts, w, h)
    if pose is not None:
        s["yaw"], s["pitch"], s["roll"] = pose
    return s


# ---------- calibracion y reglas ----------
def calibrar(muestras, n_total):
    """Devuelve (base, aviso). base = tu cara normal (mediana de cada senal).
    Si algo no cuadra, base es None y aviso dice que hacer."""
    con_pose = [m for m in muestras if not math.isnan(m["yaw"])]
    if len(muestras) < 0.6 * n_total or len(con_pose) < 0.5 * n_total:
        return None, "Casi no te veo: mira a la camara"
    base = {}
    for k in ("ear", "mar", "yaw", "pitch"):
        valores = [m[k] for m in muestras if not math.isnan(m[k])]
        base[k] = float(np.median(valores))
    if base["ear"] < 0.12:
        return None, "Abre los ojos y mira a la camara"
    if abs(base["yaw"]) > 25:
        return None, "Mira de frente a la camara"
    return base, None


def normalizar_frame(s, base, max_yaw=MAX_YAW):
    """Lo mismo que 'normalizar' del cuaderno, pero con un solo fotograma."""
    yaw = s["yaw"]
    fiable = not math.isnan(yaw) and abs(yaw) <= max_yaw
    c = math.cos(math.radians(yaw)) if not math.isnan(yaw) else math.nan   # al girar, EAR y MAR se inflan
    return {
        "ear_rel": float(s["ear"]) * c / base["ear"] if fiable else math.nan,
        "mar": float(s["mar"]) * c if fiable else math.nan,
        "yaw_rel": yaw - base["yaw"],
        "pitch_rel": s["pitch"] - base["pitch"],
    }


class Regla:
    """Cuenta cuanto tiempo lleva seguida una condicion. Es 'duracion_continua' del cuaderno,
    pero fotograma a fotograma: solo mira el pasado, que es lo unico que se tiene en directo."""

    def __init__(self, t_aviso, t_alerta, hueco):
        self.t_aviso, self.t_alerta, self.hueco = t_aviso, t_alerta, hueco
        self.inicio = None
        self.ultimo = None

    def actualizar(self, t, activo):
        if activo:
            if self.inicio is None:
                self.inicio = t
            self.ultimo = t
        elif self.inicio is not None and t - self.ultimo > self.hueco:
            self.inicio = None
        duracion = t - self.inicio if self.inicio is not None else 0.0
        if duracion >= self.t_alerta:
            return 2
        if duracion >= self.t_aviso:
            return 1
        return 0


class Monitor:
    """Junta las reglas. Recibe las senales de cada fotograma y devuelve el nivel de cada regla
    (0 = bien, 1 = AVISO, 2 = ALERTA)."""

    def __init__(self, base):
        self.base = base
        self.reglas = {nombre: Regla(*params) for nombre, params in REGLAS.items()}
        self.boca_abierta = False                                 # memoria de la histeresis
        # La histeresis de la boca tambien se adapta a la cara: si en reposo tu boca ya marca algo,
        # "cerrada" tiene que quedar por encima de ese valor. Con la boca de la prueba (0.013) salen
        # 0.10 y 0.15, los mismos del cuaderno.
        self.mar_off = max(UMBRAL_BOCA_CERRADA, base["mar"] + 0.03)
        self.mar_on = max(UMBRAL_BOCA_ABIERTA, self.mar_off + 0.05)

    def evaluar(self, t, s):
        n = normalizar_frame(s, self.base)
        abajo = n["pitch_rel"] > UMBRAL_ABAJO                     # con nan da False
        if math.isnan(n["mar"]):
            self.boca_abierta = False
        elif n["mar"] > self.mar_on:
            self.boca_abierta = True
        elif n["mar"] < self.mar_off:
            self.boca_abierta = False
        activas = {
            "ojos": bool(n["ear_rel"] < UMBRAL_OJOS) and not abajo,    # con la cabeza abajo, el EAR no cuenta
            "abajo": bool(abajo),
            "giro": bool(abs(n["yaw_rel"]) > UMBRAL_GIRO),
            "bostezo": self.boca_abierta,
            "sin cara": bool(math.isnan(s["ear"])),
        }
        niveles = {nombre: self.reglas[nombre].actualizar(t, activas[nombre]) for nombre in REGLAS}
        return niveles, n


# ---------- sonido y registro ----------
SONIDOS = {1: "/System/Library/Sounds/Ping.aiff", 2: "/System/Library/Sounds/Basso.aiff"}


class Alarma:
    """Suena al pasar a un nivel mas alto; en ALERTA se repite cada 1.5 s. Usa afplay (Mac)."""

    def __init__(self):
        self.activa = True              # la tecla m lo cambia
        self.disponible = True          # False si no hay afplay (no es un Mac)
        self.nivel_ant = 0
        self.ultimo = -99.0
        self.proceso = None

    def actualizar(self, t, nivel):
        sube = nivel > self.nivel_ant
        toca = sube or (nivel == 2 and t - self.ultimo >= 1.5)
        self.nivel_ant = nivel
        sonando = self.proceso is not None and self.proceso.poll() is None
        if sube and sonando:
            self.proceso.terminate()    # el sonido nuevo tapa al anterior
            sonando = False
        if toca and nivel > 0 and self.activa and self.disponible and not sonando:
            try:
                self.proceso = subprocess.Popen(["afplay", SONIDOS[nivel]])
            except FileNotFoundError:
                self.disponible = False
            self.ultimo = t


class Registro:
    """Apunta en un csv cuando cada regla cambia de nivel (para poder revisar la sesion luego)."""

    def __init__(self, ruta=os.path.join(RAIZ, "resultados", "registro_monitor.csv")):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        self.f = open(ruta, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["segundos", "regla", "nivel"])
        self.anterior = {}

    def anotar(self, t, niveles):
        for nombre, nivel in niveles.items():
            if nivel != self.anterior.get(nombre, 0):
                self.w.writerow([f"{t:.1f}", nombre, ("ok", "AVISO", "ALERTA")[nivel]])
                self.f.flush()
                self.anterior[nombre] = nivel

    def cerrar(self):
        self.f.close()


# ---------- camara y detector ----------
def crear_detector(ruta=os.path.join(RAIZ, "modelos", "face_landmarker.task")):
    opciones = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ruta),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opciones)


def detectar(detector, frame, ts_ms):
    """Devuelve los 478 puntos de la cara en pixeles, o None si no hay cara."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect_for_video(imagen, ts_ms)
    if not res.face_landmarks:
        return None
    h, w = frame.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)


# ---------- dibujo ----------
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
COLOR_NIVEL = {0: (80, 175, 60), 1: (40, 170, 245), 2: (60, 60, 220)}      # BGR: verde, ambar, rojo
GRIS = (95, 95, 95)
NOMBRE_NIVEL = {0: "ATENTO", 1: "AVISO", 2: "ALERTA"}


def texto(frame, s, org, escala=0.7, color=(255, 255, 255), grosor=2):
    """Texto con borde negro para que se lea sobre cualquier fondo."""
    cv2.putText(frame, s, org, FUENTE, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, FUENTE, escala, color, grosor, cv2.LINE_AA)


def dibujar_puntos(frame, pts):
    for i in OJO_DER + OJO_IZQ + BOCA:
        cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, (0, 255, 0), -1)


def dibujar_calibracion(frame, progreso, aviso):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 44), (150, 100, 40), -1)
    texto(frame, "CALIBRANDO: mira a la camara, sin hablar", (15, 31), 0.8)
    cv2.rectangle(frame, (15, 60), (w - 15, 76), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, 60), (15 + int((w - 30) * min(progreso, 1.0)), 76), (255, 200, 80), -1)
    if aviso:
        texto(frame, aviso, (15, 110), 0.7, (0, 200, 255))


def dibujar_estado(frame, niveles):
    h, w = frame.shape[:2]
    nivel = max(niveles.values())
    mensaje = NOMBRE_NIVEL[nivel]
    if nivel > 0:
        motivos = [TEXTO_MOTIVO[nombre] for nombre, v in niveles.items() if v == nivel]
        mensaje += ": " + ", ".join(motivos)
    cv2.rectangle(frame, (0, 0), (w, 44), COLOR_NIVEL[nivel], -1)
    texto(frame, mensaje, (15, 31), 0.85)
    if nivel == 2:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_NIVEL[2], 8)


def grados(x):
    return "--" if math.isnan(x) else f"{round(x):+d}"


def dibujar_numeros(frame, n):
    """Las senales respecto a tu normal: ojos 1.00 = igual que siempre, giro y cabeceo 0 = como al calibrar."""
    ear_rel = "--" if math.isnan(n["ear_rel"]) else f"{n['ear_rel']:.2f}"
    boca = "--" if math.isnan(n["mar"]) else f"{n['mar']:.2f}"
    texto(frame, f"ojos    {ear_rel}", (15, 75), 0.55)
    texto(frame, f"boca    {boca}", (15, 98), 0.55)
    texto(frame, f"giro    {grados(n['yaw_rel'])}", (15, 121), 0.55)
    texto(frame, f"cabeceo {grados(n['pitch_rel'])}", (15, 144), 0.55)


def dibujar_reglas(frame, niveles):
    """Una etiqueta por regla: gris si todo bien, ambar en AVISO, roja en ALERTA."""
    h = frame.shape[0]
    x = 15
    for nombre, nivel in niveles.items():
        (ancho, _), _ = cv2.getTextSize(nombre, FUENTE, 0.5, 1)
        color = COLOR_NIVEL[nivel] if nivel > 0 else GRIS
        cv2.rectangle(frame, (x, h - 52), (x + ancho + 16, h - 28), color, -1)
        cv2.putText(frame, nombre, (x + 8, h - 35), FUENTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += ancho + 24


def cerrar_todo(cap, detector, writer=None, registro=None):
    """Libera todo. Se puede llamar varias veces sin problema."""
    cap.release()
    detector.close()
    if writer is not None:
        writer.release()
    if registro is not None:
        registro.cerrar()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)          # en Mac, sin estas pasadas la ventana puede quedarse abierta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grabar", metavar="VIDEO.mp4", help="guarda tambien el video de la demo")
    args = parser.parse_args()

    detector = crear_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cerrar_todo(cap, detector)
        raise SystemExit("No puedo abrir la camara")

    alarma, registro = Alarma(), Registro()
    monitor, writer = None, None                    # monitor es None mientras se calibra
    muestras, n_cal, t_cal, aviso = [], 0, None, None
    t0 = time.time()
    ultimo_ts = -1
    fps, t_ant = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                     # efecto espejo (el signo del giro se invierte, da igual: usamos |giro|)
            h, w = frame.shape[:2]

            t = time.time() - t0
            ts = max(int(t * 1000), ultimo_ts + 1)         # MediaPipe exige tiempos crecientes
            ultimo_ts = ts
            pts = detectar(detector, frame, ts)
            s = medir(pts, w, h)
            if pts is not None:
                dibujar_puntos(frame, pts)

            if monitor is None:                            # fase 1: calibrar
                if t_cal is None:
                    t_cal = t
                n_cal += 1
                if not math.isnan(s["ear"]):
                    muestras.append(s)
                progreso = (t - t_cal) / SEGUNDOS_CALIBRACION
                if progreso >= 1:
                    base, aviso = calibrar(muestras, n_cal)
                    if base is None:                       # algo fallo: a repetir
                        muestras, n_cal, t_cal = [], 0, t
                    else:
                        monitor = Monitor(base)
                        print("Tu cara normal:", {k: round(v, 3) for k, v in base.items()})
                        if args.grabar:
                            fps_real = n_cal / (t - t_cal)
                            writer = cv2.VideoWriter(args.grabar, cv2.VideoWriter_fourcc(*"mp4v"), fps_real, (w, h))
                dibujar_calibracion(frame, progreso, aviso)
            else:                                          # fase 2: vigilar
                niveles, n = monitor.evaluar(t, s)
                registro.anotar(t, niveles)
                alarma.actualizar(t, max(niveles.values()))
                dibujar_estado(frame, niveles)
                dibujar_numeros(frame, n)
                dibujar_reglas(frame, niveles)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 / max(ahora - t_ant, 1e-6)
            t_ant = ahora
            texto(frame, f"{fps:.0f} fps   sonido: {'si' if alarma.activa else 'no'}", (w - 230, h - 15), 0.5)
            if writer is not None:
                writer.write(frame)

            cv2.imshow("Monitor del conductor", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):                    # ESC o q
                break
            if tecla == ord("m"):
                alarma.activa = not alarma.activa
            if tecla == ord("c"):                          # volver a calibrar
                monitor, muestras, n_cal, t_cal, aviso = None, [], 0, None, None
                alarma.nivel_ant = 0
    except KeyboardInterrupt:
        print("Interrumpido con Ctrl+C")
    finally:
        cerrar_todo(cap, detector, writer, registro)       # se ejecuta SIEMPRE, haya error o no
        print("Camara liberada.")
        if args.grabar:
            print("Video guardado en", args.grabar)


if __name__ == "__main__":
    main()
