"""
PROTOTIPO A: Reto no te rias  (version 1: espejo en tiempo real)

Uso:  python reto_no_te_rias.py        (ESC para salir)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow import keras
import random 

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # carpeta Practica5_VC

CLASES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IDX_HAPPY = CLASES.index("happy")
OJO_A = [33, 133, 159, 145]
OJO_B = [263, 362, 386, 374]
ALFA = 0.7          # suavizado: cuanto pesa el pasado (0 = nada, 0.9 = muy lento)


# --- Reglas del juego ---
UMBRAL_RISA = 0.5   # probabilidad suavizada de 'happy' a partir de la cual cuenta como risa
SUBE = 2.0          # cuanto sube el medidor por segundo mientras te ries
BAJA = 1.0          # cuanto baja por segundo mientras aguantas
CUENTA_ATRAS = 3    # segundos de cuenta atras
MAX_SIN_CARA = 2.0  # segundos sin cara antes de perder
ESPERANDO, CUENTA, JUGANDO, PERDIDO = range(4)

SEG_MEME = 6        # segundos que se ve cada meme
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")

def recortar_cara(frame, pts, tam=48, margen=1.1):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a = pts[OJO_A].mean(axis=0)
    b = pts[OJO_B].mean(axis=0)
    angulo = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    medio = (float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2))
    R = cv2.getRotationMatrix2D(medio, angulo, 1.0)
    gris_girada = cv2.warpAffine(gris, R, (gris.shape[1], gris.shape[0]),
                                 borderMode=cv2.BORDER_REPLICATE)
    pts_girados = pts @ R[:, :2].T + R[:, 2]
    x1, y1 = pts_girados.min(axis=0)
    x2, y2 = pts_girados.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    lado = max(x2 - x1, y2 - y1) * margen
    pad = int(lado)
    gris_pad = cv2.copyMakeBorder(gris_girada, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    l = int(round(lado))
    x0 = int(round(cx - lado / 2)) + pad
    y0 = int(round(cy - lado / 2)) + pad
    recorte = gris_pad[y0:y0 + l, x0:x0 + l]
    return cv2.resize(recorte, (tam, tam), interpolation=cv2.INTER_AREA)


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


def texto(frame, s, org, escala=0.7, color=(255, 255, 255), grosor=2):
    """Texto con borde negro para que se lea sobre cualquier fondo."""
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar_info(frame, probs, recorte):
    h, w = frame.shape[:2]
    top = int(np.argmax(probs))
    texto(frame, f"{CLASES[top]} {probs[top] * 100:.0f}%", (15, 35))
    p = float(probs[IDX_HAPPY])
    cv2.rectangle(frame, (15, 55), (215, 75), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, 55), (15 + int(200 * p), 75), (0, 200, 255), -1)
    texto(frame, f"risa: {p * 100:.0f}%", (15, 100), 0.6)
    mini = cv2.cvtColor(cv2.resize(recorte, (96, 96), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)
    frame[10:106, w - 106:w - 10] = mini

def texto_centrado(frame, s, y, escala=0.8, color=(255, 255, 255), grosor=2):
    (ancho, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    texto(frame, s, ((frame.shape[1] - ancho) // 2, y), escala, color, grosor)


def barra_medidor(frame, nivel):
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = 15, w - 15, h - 50, h - 22
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), -1)
    color = (0, int(255 * (1 - nivel)), int(255 * nivel))        # BGR: de verde a rojo
    cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * nivel), y1), color, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    texto(frame, "medidor de risa", (x0, y0 - 8), 0.6)


def pantalla_perdido(foto, tiempo, mejor, motivo):
    h, w = foto.shape[:2]
    rojo = np.full_like(foto, (0, 0, 200))
    vista = cv2.addWeighted(foto, 0.7, rojo, 0.3, 0)        # tinte rojo sobre la foto del momento
    cv2.rectangle(vista, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
    texto_centrado(vista, "HAS PERDIDO!", 75, 2.0, (255, 255, 255), 5)
    texto_centrado(vista, "Se te escapo la risa" if motivo == "risa" else "Te saliste de la camara", 115, 0.9)
    texto_centrado(vista, f"Aguantaste {tiempo:.1f} s", h - 95, 1.2, (255, 255, 255), 3)
    texto_centrado(vista, f"Mejor marca: {mejor:.1f} s", h - 60, 0.8, (0, 200, 255))
    texto_centrado(vista, "ESPACIO para reintentar", h - 25, 0.7)
    return vista


def ajustar_a_panel(img, ancho, alto):
    h, w = img.shape[:2]
    esc = min(ancho / w, alto / h)
    interp = cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR
    nueva = cv2.resize(img, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=interp)
    panel = np.full((alto, ancho, 3), 25, np.uint8)
    y0 = (alto - nueva.shape[0]) // 2
    x0 = (ancho - nueva.shape[1]) // 2
    panel[y0:y0 + nueva.shape[0], x0:x0 + nueva.shape[1]] = nueva
    return panel

def panel_texto(ancho, alto, lineas):
    panel = np.full((alto, ancho, 3), 25, np.uint8)
    y = alto // 2 - 20 * (len(lineas) - 1)
    for s in lineas:
        texto_centrado(panel, s, y, 0.9)
        y += 40
    return panel

def panel_derecho(juego, memes, ancho, alto):
    if juego.estado == JUGANDO and juego.meme is not None:
        return memes.panel(juego.meme, ancho, alto)
    if juego.estado == PERDIDO and juego.motivo == "risa" and juego.meme is not None:
        panel = memes.panel(juego.meme, ancho, alto).copy()
        texto_centrado(panel, "EL CULPABLE", 40, 1.0, (0, 200, 255), 3)
        return panel
    if len(memes) == 0:
        return panel_texto(ancho, alto, ["No hay memes", "Pon imagenes en la carpeta", "contenido/"])
    if juego.estado == PERDIDO:
        return panel_texto(ancho, alto, ["Partida terminada"])
    return panel_texto(ancho, alto, ["Aguanta sin reirte", "Los memes empiezan", "al acabar la cuenta atras"])


class Juego:
    """Guarda el estado de la partida y aplica las reglas."""

    def __init__(self, n_memes=0):
        self.estado = ESPERANDO
        self.t_estado = 0.0      # momento en que empezo el estado actual
        self.nivel = 0.0         # medidor de risa, de 0 a 1
        self.tiempo = 0.0        # segundos aguantados en la partida actual
        self.mejor = 0.0
        self.t_sin_cara = 0.0
        self.motivo = ""
        self.foto = None         # fotograma en el que se perdio

        self.n_memes = n_memes
        self.orden = []          # orden aleatorio de los memes en esta partida
        self.meme = None         # meme que se ve ahora
        self.risas = {}          # cuantas veces ha ganado cada meme

    def pulsar_espacio(self, ahora):
        if self.estado in (ESPERANDO, PERDIDO):
            self.estado, self.t_estado = CUENTA, ahora

    def actualizar(self, ahora, dt, hay_cara, p_risa, frame):
        if self.estado == CUENTA:
            if not hay_cara:
                self.estado = ESPERANDO
            elif ahora - self.t_estado >= CUENTA_ATRAS:
                self.estado, self.t_estado = JUGANDO, ahora
                self.nivel, self.t_sin_cara, self.tiempo = 0.0, 0.0, 0.0
                self.orden = random.sample(range(self.n_memes), self.n_memes)

        elif self.estado == JUGANDO:
            self.tiempo = ahora - self.t_estado

            if self.n_memes > 0:
                self.meme = self.orden[int(self.tiempo // SEG_MEME) % self.n_memes]

            if hay_cara:
                self.t_sin_cara = 0.0
                if p_risa >= UMBRAL_RISA:
                    self.nivel += dt * SUBE
                else:
                    self.nivel -= dt * BAJA
                self.nivel = min(max(self.nivel, 0.0), 1.0)
            else:
                self.t_sin_cara += dt
            if self.nivel >= 1.0 or self.t_sin_cara > MAX_SIN_CARA:
                self.motivo = "risa" if self.nivel >= 1.0 else "cara"
                self.mejor = max(self.mejor, self.tiempo)
                self.foto = frame.copy()

                if self.motivo == "risa" and self.meme is not None:
                    self.risas[self.meme] = self.risas.get(self.meme, 0) + 1
                self.estado, self.t_estado = PERDIDO, ahora

    def dibujar(self, frame, ahora, hay_cara):
        h, w = frame.shape[:2]
        if self.estado == PERDIDO:
            return pantalla_perdido(self.foto, self.tiempo, self.mejor, self.motivo)
        if self.estado == ESPERANDO:
            texto_centrado(frame, "Pulsa ESPACIO para empezar", h - 30, 0.9, (0, 200, 255))
        elif self.estado == CUENTA:
            restante = CUENTA_ATRAS - (ahora - self.t_estado)
            texto_centrado(frame, str(max(1, int(np.ceil(restante)))), h // 2 + 60, 6, (0, 200, 255), 12)
            texto_centrado(frame, "Prepara tu cara seria...", h - 30, 0.9)
        elif self.estado == JUGANDO:
            texto_centrado(frame, f"{self.tiempo:.1f} s", 50, 1.4, (255, 255, 255), 3)
            texto_centrado(frame, f"mejor: {self.mejor:.1f} s", 85, 0.6)
            barra_medidor(frame, self.nivel)
            if not hay_cara:
                texto_centrado(frame, "Vuelve a la camara!", h // 2, 1.0, (0, 200, 255), 3)
        return frame
    
class Memes:
    """Carga las imagenes de la carpeta y las prepara para el panel."""

    def __init__(self, carpeta=os.path.join(RAIZ, "contenido")):
        self.nombres, self.imagenes, self.cache = [], [], {}
        if not os.path.isdir(carpeta):
            return
        for nombre in sorted(os.listdir(carpeta)):
            if nombre.lower().endswith(EXTENSIONES):
                img = cv2.imread(os.path.join(carpeta, nombre))
                if img is not None:
                    self.nombres.append(nombre)
                    self.imagenes.append(img)

    def __len__(self):
        return len(self.imagenes)

    def panel(self, idx, ancho, alto):
        clave = (idx, ancho, alto)
        if clave not in self.cache:
            self.cache[clave] = ajustar_a_panel(self.imagenes[idx], ancho, alto)
        return self.cache[clave]
    
def main():
    modelo = keras.models.load_model(os.path.join(RAIZ, "modelos", "emociones.keras"))
    detector = crear_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No puedo abrir la camara")

    memes = Memes()
    print(f"Memes cargados: {len(memes)}")
    juego = Juego(len(memes))

    probs, debug = None, False
    t0 = time.time()
    ultimo_ts = -1
    t_ant = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                         # efecto espejo
        ahora = time.time()
        dt = ahora - t_ant
        t_ant = ahora

        # 1) De la imagen a la probabilidad de risa
        ts = max(int((ahora - t0) * 1000), ultimo_ts + 1)  # MediaPipe exige tiempos crecientes
        ultimo_ts = ts
        pts = detectar(detector, frame, ts)
        hay_cara = pts is not None
        if hay_cara:
            recorte = recortar_cara(frame, pts)
            x = recorte.astype("float32")[None, :, :, None]
            p = modelo(x, training=False).numpy()[0]
            probs = p if probs is None else ALFA * probs + (1 - ALFA) * p
            p_risa = float(probs[IDX_HAPPY])
        else:
            probs, p_risa = None, 0.0

        # 2) Reglas del juego y dibujo
        juego.actualizar(ahora, dt, hay_cara, p_risa, frame)
        if not hay_cara and juego.estado != PERDIDO:
            texto(frame, "No veo ninguna cara", (15, 35), 0.8, (0, 200, 255))
        if debug and hay_cara:
            dibujar_info(frame, probs, recorte)
        frame = juego.dibujar(frame, ahora, hay_cara)

        alto = frame.shape[0]
        lienzo = np.hstack([frame, panel_derecho(juego, memes, alto, alto)])
        cv2.imshow("Reto no te rias", lienzo)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:                                        # ESC
            break
        if k == 32:                                        # ESPACIO
            juego.pulsar_espacio(ahora)
        if k == ord("d"):                                  # d: mostrar/ocultar datos tecnicos
            debug = not debug

    cap.release()
    detector.close()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    if juego.risas:
        print("Memes que te han hecho perder:")
        for idx in sorted(juego.risas, key=juego.risas.get, reverse=True):
            print(f"  {memes.nombres[idx]}: {juego.risas[idx]} veces")

if __name__ == "__main__":
    main()