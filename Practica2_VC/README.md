# Práctica 2: Funciones básicas de OpenCV. Documentación

Autora 1: Solmaire Basulto Santana  
Autor 2: Gerad Leopoldo Guzmán Hernández

---

### 1. Descripción del Trabajo
Este repositorio contiene el cuaderno Jupyter con la resolución de las tareas de la segunda práctica de Visión por Computador. El objetivo principal ha sido aplicar diferentes técnicas básicas de procesamiento de imagen, como la detección de bordes, el umbralizado y el análisis matemático de los píxeles de una matriz, finalizando en el diseño de un sistema interactivo en tiempo real utilizando la cámara web.

El trabajo se divide en  3 tareas principales:    

Tarea 1: Análisis de perfiles con Canny.   
Tarea 2: Análisis comparativo con Sobel.   
Tarea 3: Demostrador interactivo (Virtual Drums)
   
### 2. Paquetes necesarios
Podemos hacer uso del mismo environment de la primera práctica, asegurándonos de tener instaladas las librerías de visión, cálculo y audio.

*Nota:* El código incluye la instrucción `cv2.waitKey(1)` para evitar el cuelgue del kernel que se produce en el entorno de Mac al usar `imshow`. Además, se han comentado las lineas en donde se hace uso de la función `imshow` ya que Mac ignora estas lineas.

  
### 3. Análisis y Resolución de Tareas  

#### Tarea 1: Análisis de perfiles por filas y umbralización con Canny
Objetivo: Realizar la cuenta de píxeles blancos por filas. Determinar el valor máximo y resaltar las filas que tengan un número de píxeles mayor o igual al 90% de este máximo sobre la imagen de Canny.  

Desarrollo:  

Para esta tarea, se partió de la imagen original (mandril.jpg) convertida a escala de grises. Se aplicó el algoritmo de Canny `cv2.Canny` para detectar los bordes, obteniendo una imagen binaria.

En lugar de recorrer la matriz de la imagen con bucles for (lo cual es ineficiente en Python), se utilizó la función `cv2.reduce` de OpenCV. Esta función permite empaquetar los datos de la matriz.

Una vez obtenida la lista con la suma de cada fila (fila_counts), se calculó el valor máximo `maxfil`. Posteriormente, se iteró sobre esta lista para encontrar los índices de las filas que cumplían la condición (fila_counts[i] >= maxfil * 0.90), almacenándolos en el array pos.
Finalmente, se convirtió la imagen Canny a formato RGB `cv2.cvtColor` para poder dibujar en color, y se trazó una línea azul `cv2.line` cruzando todo el ancho de la imagen en las coordenadas y correspondientes a las filas almacenadas en pos.   

  
#### Tarea 2: Análisis de perfiles por filas y umbralización con Canny
Objetivo: Aplicar el operador Sobel para la detección de bordes, umbralizar el resultado y repetir el proceso de conteo de píxeles (esta vez tanto por filas como por columnas). Comparar gráficamente los resultados obtenidos entre Sobel y Canny.

Desarrollo:  

El procedimiento comenzó suavizando la imagen en escala de grises con un filtro Gaussiano `cv2.GaussianBlur` para reducir el ruido. A continuación, se calculó el gradiente de la imagen utilizando el operador Sobel en ambas direcciones (X e Y) mediante derivadas de 64 bits (cv2.CV_64F), sumando posteriormente ambos resultados y convirtiéndolos a 8 bits con `cv2.convertScaleAbs`.
El resultado del operador Sobel presenta diferentes escalas de grises dependiendo de la intensidad del gradiente. Para poder realizar el conteo de píxeles igual que en la tarea anterior, fue necesario binarizar la imagen usando `cv2.threshold` con un umbral estricto (se definió en 190).

Se aplicó nuevamente `cv2.reduce`, añadiendo `.flatten()` para aplanar el array devuelto por OpenCV y evitar errores de dimensionalidad. Tras calcular los máximos (maxfil, maxcol) y guardar las posiciones que superaban el umbral del 90%, se dibujaron las líneas horizontales y verticales sobre la imagen original.`


  
#### Tarea 3: Demostrador interactivo (Virtual Drums)
Objetivo: Diseñar un demostrador interactivo en tiempo real que reinterprete el procesamiento de imagen inspirado en instalaciones como Virtual Air Guitar o My little piece of privacy.

Desarrollo:  

Básicamente, la idea es dividir la parte inferior de la pantalla en cuatro zonas independientes, donde cada una reproduce un sonido distinto de la batería. Para poder interactuar con el programa y tocar.

Para que el procesamiento funcione bien y rápido, se ha dividido el código en dos partes principales:

1. Buscar el color (Filtro HSV):
Primero, se convierte el fotograma de la cámara al espacio de color HSV `cv2.cvtColor`. Esto es importante porque el HSV resiste mucho mejor los cambios de luz de la habitación que el RGB normal. Luego, con la función `cv2.inRange` y pasándole los límites del color rosa (rosa_bajo y rosa_alto), se consigue una máscara binaria. En esta máscara, todo lo de color rosa sale en blanco y el resto de la habitación se queda en negro puro, por lo que el programa ya no se confunde con el fondo.

2. Detectar el golpe real (Diferencia temporal):
En esta parte se encontró un problema: si se dejaba "la baqueta" quieta dentro de una zona, el programa veía el color rosa y reproducía el sonido sin parar. Para evitarlo, se necesitó detectar solo el movimiento. Se solucionó usando `cv2.absdiff` para restar la máscara rosa actual con la del fotograma anterior (pframe). Así, el programa solo pinta de blanco los píxeles rosas que se están desplazando.

Una vez se obtiene esa imagen del movimiento, se recorta en cuatro cuadrantes. Luego se utiliza `cv2.countNonZero` en cada trozo y si hay suficientes píxeles blancos moviéndose (superan el umbral_golpe), se hace sonar el instrumento correspondiente con la librería `pygame`.  

  
Problemas de lag y feedback visual
Al principio, para que el sonido no se repitiera, se intentó poner retardos, pero eso congelaba la cámara y daba mucho lag. La solución final fue usar "banderas de estado" (tocando_platillo = False). Gracias a esto, el sonido salta justo en el instante del golpe y bloquea la repetición hasta que se detiene el movimiento, lo que permite dar golpes muy rápidos sin que la cámara se trabe.
Por último, para que la experiencia fuera más interactiva al mirar la pantalla, se añadió un efecto visual con `cv2.addWeighted` que ilumina el cuadrante con un color semitransparente justo en el momento en el que detecta el golpe.  

### 4. Bibliografía y Fuentes Consultadas

Documentación Oficial de OpenCV:
- [Detect the edges of the image by using a Canny detector](https://docs.opencv.org/4.13.0/d9/db0/tutorial_hough_lines.html#autotoc_md697)
- [Operations on arrays](https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981)
- [Image filtering](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)

