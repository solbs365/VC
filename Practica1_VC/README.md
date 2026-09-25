# Práctica 1: Introducción a OpenCV. Documentación

Autora 1: Solmaire Basulto Santana  
Autor 2: Gerad Leopoldo Guzmán Hernández

---

### 1. Descripción del Trabajo
Este repositorio contiene el cuaderno Jupyter con la resolución de las tareas de la primera práctica de Visión por Computador. El objetivo principal ha sido familiarizarse con la manipulación de matrices en NumPy, la lectura/escritura de imágenes y vídeo, y el uso de las funciones básicas de dibujo y procesamiento de OpenCV.

El trabajo se divide en cuatro tareas principales:
1. Creación de un tablero de ajedrez (Manual vs IA).
2. Composición artística basada en el estilo de Piet Mondrian.
3. Procesamiento de vídeo en tiempo real para detección de píxeles extremos.
4. Creación de un filtro estilo "Pop Art".

### 2. Requisitos e Instalación
Para la correcta ejecución del cuaderno, el entorno debe contar con las siguientes librerías instaladas:
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`

*Nota:* El código incluye la instrucción `cv2.waitKey(1)` para evitar el cuelgue del kernel que se produce en el entorno de Mac al usar `imshow`. Además, se han comentado las lineas en donde se hace uso de la función `imshow` ya que Mac ignora estas lineas.

### 3. Análisis y Resolución de Tareas

#### Tarea 1: Tablero de Ajedrez (Manual vs IA)
Se implementaron dos versiones del tablero para observar las diferencias de optimización:

- Versión Manual: Basada en bucles anidados (`for`). Es funcional pero ineficiente al tratar con imágenes grandes.
- Versión IA: El asistente propuso una solución basada en slicing de matrices de NumPy. Esta versión calcula los bloques de coordenadas de inicio y fin, y asigna el valor `255` directamente a porciones completas de la matriz. A nivel de rendimiento y limpieza de código, la versión vectorizada (IA) es netamente superior y más escalable.

#### Tarea 2: Estilo Mondrian
Se construyó un lienzo blanco con (`color_img[:,:,:] = 255`) y se utilizaron las funciones geométricas de openCV `cv2.line` y `cv2.rectangle` (con grosor `-1` para el relleno de la figura). El ejercicio sirvió para comprender el sistema de coordenadas de OpenCV, donde el origen `(0,0)` se sitúa en la esquina superior izquierda.

#### Tarea 3: Píxel más claro y oscuro
Se utilizó la función `cv2.minMaxLoc` tras convertir el fotograma a escala de grises. 

- Justificación: A pesar de procesar miles de píxeles por fotograma, se obtiene gran fluidez debido a que `minMaxLoc` está implementada a bajo nivel (C++) y altamente optimizada. 
- Posible Aceleración: En caso de sufrir saltos en equipos más lentos, la solución más eficaz sería reducir la dimensionalidad realizando un `cv2.resize()` del `frame_gris` antes de buscar los mínimos y máximos, reduciendo exponencialmente el área de búsqueda.

#### Tarea 4: Pop Art Personalizado (Mosaico 3x3 y Máscaras Booleanas)
Se implementó un mosaico de 9 pantallas (3x3). 
Para lograr el efecto cómic, no se realizó la separación de canales BGR. En su lugar se emplearon las siguientes técnicas:

1. Cuantización de Color: Se pasó la imagen a escala de grises y se aplicó la operación lógica `(gris > 100) * 255`. Esto convierte la imagen en un mapa binario, eliminando los degradados.
2. Máscaras Booleanas: Se utilizaron matrices lógicas (ej. `gris_plano == 255`) para inyectar directamente colores exactos (formato BGR). Esto ofrece un control absoluto del color final, logrando el efecto cómic buscado.


### 4. Bibliografía y Fuentes Consultadas

Documentación Oficial de OpenCV:
- [Image file reading and writing (imread)](https://docs.opencv.org/4.13.0/d4/da8/group__imgcodecs.html#gaffb68fce322c6e52841d7d9357b9ad2d)
- [High-level GUI (imshow)](https://docs.opencv.org/4.13.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)
- [Getting Started with Videos](https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html)
- [Drawing Functions in OpenCV](https://docs.opencv.org/4.13.0/dc/da5/tutorial_py_drawing_functions.html)
- [Color Space Conversions](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html)
- [Operations on arrays (minMaxLoc)](https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html#gab473bf2eb6d14ff97e89b355dac20707)

Artículos y tutoriales externos:
- Inspiración para la asignación de colores planos en la tarea 4 (Pop Art): [Creating Pop Art using OpenCV and Python (Analytics Link)](https://www.analytics-link.com/post/2019/07/11/creating-pop-art-using-opencv-and-python)
