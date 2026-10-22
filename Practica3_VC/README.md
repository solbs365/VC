# Práctica 3: Detección y reconocimiento de formas. Documentación

Autora 1: [Solmaire Basulto Santana](https://github.com/solbs365)  
Autor 2: [Gerad Leopoldo Guzmán Hernández](https://github.com/gerad367)

---

### 1. Descripción del Trabajo
Este repositorio contiene el cuaderno Jupyter con la resolución de las tareas de la tercera práctica de Visión por Computador. El objetivo ha sido sacar información geométrica de los objetos que aparecen en una imagen (área, perímetro, forma del contorno...) para poder caracterizarlos y reconocerlos después de forma automática. Para ello se ha trabajado con el umbralizado, la detección de contornos y la transformada de Hough, y en ninguna de las tareas se ha usado aprendizaje profundo.

El trabajo se divide en 2 tareas principales:

- Tarea 1: Contador de dinero con monedas (imagen ideal y demostrador en vivo).
- Tarea 2: Clasificación de microplásticos (fragmentos, pellets y alquitrán).

### 2. Paquetes necesarios
Se puede usar el mismo environment de las prácticas anteriores. Lo único nuevo son `scikit-learn` y `seaborn`, que se usan para calcular las métricas y dibujar la matriz de confusión:

```
pip install scikit-learn seaborn
```

Además se importan `cv2`, `numpy`, `matplotlib`, `pandas`, `csv` y `math`.

*Nota:* Igual que en la práctica anterior, el código incluye `cv2.waitKey(1)` después de `cv2.destroyAllWindows()` para evitar el cuelgue del kernel en Mac.

### 3. Análisis y Resolución de Tareas

#### Tarea 1: Contador de monedas
Objetivo: calcular la cantidad de dinero que hay en una imagen de monedas. Primero con la imagen ideal que se proporciona (`Monedas.jpg`) y después en vivo con la cámara web.

**Imagen ideal**

Desarrollo:

Se partió de lo que sugiere el enunciado: si se sabe lo que mide una moneda en la realidad, se puede sacar la relación píxeles/milímetros y con ella el tamaño real de todas las demás.

Primero se localizan las monedas con la transformada de Hough (`cv2.HoughCircles`) sobre la imagen en grises suavizada con `cv2.medianBlur`, y se dibujan los círculos encontrados. Luego, con `cv2.setMouseCallback`, se hace clic sobre la moneda de 1€ y se busca qué círculo contiene el punto pulsado (con Pitágoras, comprobando que la distancia al centro es menor que el radio). El diámetro de esa moneda en píxeles junto con sus 23.25 mm reales da la relación `pix_to_mm`. A partir de ahí, cada círculo se pasa a milímetros y se le asigna la moneda de la tabla `coin_size` cuyo diámetro se parece más.

Un detalle: en el enunciado el diámetro de la moneda de 50 céntimos aparece como 24.35 mm, pero el valor real es 24.25 mm, que es el que usamos en la tabla.

Primer intento y problema:

En la primera versión el diámetro se calculaba directamente con el radio que devuelve Hough (`radius * 2`). El resultado fue que tres monedas se clasificaban mal, y la de 2 céntimos era una de ellas:

| Moneda real | Diámetro real (mm) | Medido con Hough (mm) | Clasificada como |
| :---: | :---: | :---: | :---: |
| 5 cts | 21.25 | 20.08 | 10 cts |
| 10 cts | 19.75 | 20.87 | 5 cts |
| 2 cts | 18.75 | 19.55 | 10 cts |

El total salía 3.96€ cuando en la imagen hay 3.88€. Se parecía solo porque unos errores compensaban a otros.

Mirando los números nos dimos cuenta de que el radio que da Hough no es nada preciso. Por ejemplo, la moneda de 10 cts salía más grande que la de 5 cts (radios de 79.3 y 76.2 píxeles) cuando en la realidad es justo al revés. Las monedas tienen varios bordes (el canto, el aro interior, la sombra) y Hough se puede quedar con cualquiera de ellos. Además, como todo se calibra con la moneda de 1€, si su radio sale un poco pequeño todas las demás salen infladas. La de 2 cts es la que más lo sufre, porque solo hay 1 mm de diferencia con la de 10 cts: la frontera está en 19.25 mm y la nuestra salió en 19.55 mm.

Solución:

Se dejó Hough solo para saber **dónde** está cada moneda, y el tamaño se pasó a medir con el contorno. Como en la imagen ideal el fondo es blanco, se umbraliza la imagen en grises (`cv2.threshold` con 200 y `THRESH_BINARY_INV`), de forma que todo lo que es más oscuro es moneda. Después se hace un cierre morfológico (`cv2.morphologyEx`) y se sacan los contornos externos con `cv2.findContours`.

La función `measure_diameter` busca el contorno que contiene el centro del círculo de Hough (`cv2.pointPolygonTest`) y calcula el diámetro a partir de su área: `d = sqrt(4 * A / π)`. Como el área sale de muchos píxeles, el error es mucho menor que con el radio de Hough. Con este cambio las 8 monedas se clasifican bien y el total es 3.88€.

| Círculos detectados con Hough | Salida con el primer intento | Salida final |
| :---: | :---: | :---: |
| <img src="img/tarea1_1_hough.jpg" width="250"/> | <img src="img/tarea1_2_salida_error.png" width="250"/> | <img src="img/tarea1_3_salida_final.png" width="250"/> |

**Demostrador en vivo**

Desarrollo:

Se reutilizó el mismo procedimiento dentro del bucle de la cámara (`cv2.VideoCapture`), poniendo las monedas sobre un folio blanco. En cada fotograma se pasa a grises, se calculan los contornos con el mismo umbral y se buscan los círculos con Hough. Al hacer clic en la moneda de 1€ se calcula `pix_to_mm`, y desde ese momento el total se recalcula en cada fotograma y se escribe en pantalla con `cv2.putText`.

Respecto a la versión de la imagen ideal hubo que cambiar algunas cosas:

- `count_money` ahora devuelve el total en lugar de imprimirlo. Si no, escribía decenas de líneas por segundo en la consola.
- `count_money` comprueba que haya círculos (`circ is not None`), porque en vivo hay fotogramas en los que Hough no encuentra ninguno y se rompía.
- Los contornos se calculan con la imagen en grises antes de dibujar los círculos verdes, para que lo dibujado no cambie la medida.

¿Funciona correctamente? ¿Se observan problemas?

Con las monedas sobre el folio blanco y buena luz funciona bien. Los problemas principales son:

- **Depende de la iluminación.** El umbral está fijo en 200, así que el folio tiene que verse casi blanco en toda la imagen. Si se ve grisáceo (sombras, poca luz, esquinas oscuras), todo el folio pasa a ser un único contorno enorme y el total sale mal.
- **Hough no siempre detecta todas las monedas.** Alguna moneda pequeña puede dejar de detectarse en algunos fotogramas y el total cambia de un fotograma a otro.
- **Objetos que no son monedas.** Ahora mismo cualquier círculo que detecta Hough se cuenta como la moneda más parecida (`min_diference = 1000`), así que un objeto redondo sumaría dinero.
- **Monedas solapadas.** Si dos monedas se tocan, sus contornos se unen en uno solo y el diámetro sale demasiado grande.

| Demostrador sin calibrar | Demostrador con el total |
| :---: | :---: |
| <img src="img/tarea1_4_demo_sin_calibrar.jpg" width="300"/> | <img src="img/tarea1_5_demo_total.jpg" width="300"/> |

#### Demostración de uso


https://github.com/user-attachments/assets/717124f9-7895-4717-ab6b-e7ede48aff07



#### Tarea 2: Clasificación de microplásticos
Objetivo: extraer las partículas de una imagen de test (`MPs_test.jpg`) y clasificarlas en tres clases: fragmentos (FRA), pellets (PEL) y alquitrán (TAR). Solo se pueden usar las imágenes de entrenamiento para construir el clasificador, la imagen de test se usa únicamente para evaluar con sus anotaciones (`MPs_test_bbs.csv`). No se usa aprendizaje profundo ni para segmentar ni para clasificar.

Desarrollo:

El procedimiento tiene cuatro partes: segmentar, calcular características, clasificar y evaluar.

**1. Segmentación**

Según el enunciado, el umbralizado es delicado para separar las partículas del fondo, así que se probó a no depender de un único valor de gris:

- Se suaviza la imagen con `cv2.GaussianBlur`.
- Se estima el fondo con una mediana muy grande (`cv2.medianBlur` con tamaño 31 sobre la imagen reducida al 10 %). Así se compensa que la luz no sea uniforme y las partículas no cuentan como fondo.
- Se calcula la diferencia con el fondo (`cv2.absdiff`) y se coge la mayor entre los tres canales de color. De esta forma se detecta una partícula tanto si es más clara, más oscura o simplemente de otro color que el fondo.
- Se umbraliza esa diferencia con Otsu, pero limitando el valor entre 10 y 25. Sin ese límite, Otsu se dejaba llevar por el alquitrán, que es muy oscuro, y se perdían las partículas claras.
- Se limpia con una apertura y un cierre morfológicos, se sacan los contornos externos (`cv2.RETR_EXTERNAL`) y se descartan los de menos de 100 píxeles de área, que son ruido.

**2. Características**

Para cada contorno se calculan las características geométricas del artículo SMACC, más dos de apariencia:

| Característica | Cómo se calcula |
| :--- | :--- |
| Área y perímetro | `cv2.contourArea` y `cv2.arcLength` |
| Compacidad | perímetro² / área |
| Relación con el contenedor | área de la partícula / área del rectángulo de área mínima (`cv2.minAreaRect`) |
| Relación ancho/alto | lado corto / lado largo de ese rectángulo |
| Relación de ejes de la elipse | eje menor / eje mayor de `cv2.fitEllipse` |
| Relación de distancias | distancia mínima / máxima del centroide (`cv2.moments`) al contorno |
| Brillo y saturación | media dentro de la partícula en gris y en el canal S de HSV |

Se usó el rectángulo girado y no el vertical para que la característica no dependa de cómo esté orientada la partícula. El brillo y la saturación se añadieron para ayudar con el alquitrán (oscuro) y con los fragmentos de colores.

**3. Clasificador**

Las imágenes de entrenamiento (`FRA.png`, `PEL.png` y `TAR.png`) tienen partículas de una sola clase cada una, así que cada partícula que se segmenta ya viene etiquetada. Se calculan sus características y se normalizan con la media y la desviación del conjunto de entrenamiento, para que ninguna pese más que otra solo por tener números más grandes (el área, por ejemplo).

Para clasificar una partícula nueva se hizo un clasificador de los k vecinos más cercanos programado a mano con `numpy`: se calcula la distancia a todas las muestras de entrenamiento, se cogen las 3 más cercanas y se asigna la clase que más se repite. No se usa ningún modelo de `scikit-learn`, esa librería solo se utiliza para las métricas.

**4. Evaluación**

Se segmenta la imagen de test y se calculan las características igual que en el entrenamiento. Para comparar con las anotaciones, a cada caja del csv se le asigna la partícula segmentada cuyo centro cae dentro de la caja (si hay varias, la más grande). Las cajas que no tienen ninguna partícula se cuentan aparte como "sin detectar", y con el resto se calculan las métricas y la matriz de confusión.

| Segmentación de las imágenes de entrenamiento | Segmentación del test |
| :---: | :---: |
| <img src="img/tarea2_1_entrenamiento.jpg" width="400"/> | <img src="img/tarea2_2_test_segmentado.jpg" width="300"/> |

Partículas segmentadas en cada imagen de entrenamiento (en el enunciado se dice que hay unas 80 de fragmentos, 55 de pellets y 54 de alquitrán):

| FRA | PEL | TAR |
| :---: | :---: | :---: |
| 11 | 11 | 8 |

Resultados en la imagen de test:

- Partículas segmentadas: 90
- Anotaciones sin partícula segmentada: 36

| Accuracy | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: |
| 0.79 | 0.80 | 0,79 | 0.78 |

Las tres últimas métricas están calculadas con `average='weighted'`, es decir, promediando las clases según el número de muestras que tiene cada una.

| Matriz de confusión | Resultado sobre la imagen |
| :---: | :---: |
| <img width="350" alt="image" src="https://github.com/user-attachments/assets/ab1ef488-648c-4b01-87ec-e281d5c93c0e" /> | <img width="350" alt="image" src="https://github.com/user-attachments/assets/8cee6a6e-2b1b-4924-a577-b63f00c35cbb" /> |

En la matriz, las filas son la clase anotada y las columnas la predicha, así que la diagonal son los aciertos. Las confusiones más habituales son entre FRA y PEL.

Problemas y mejoras:

- Si dos partículas están pegadas, la segmentación las junta en un solo contorno y esa anotación no se clasifica bien.
- En el entrenamiento no se descartan las partículas que quedan cortadas por el borde de la imagen, y esas tienen una forma que no es la real.
- El clasificador depende mucho de la segmentación: si el contorno no es bueno, las características tampoco.
- Una mejora posible sería separar las partículas unidas con *watershed*, o probar más características de color o textura.

### 4. Uso de herramientas de IA
Para esta práctica hemos utilizado Claude (Anthropic) como asistente:

- **Tarea 1:** nos ayudó a entender por qué la moneda de 2 cts se clasificaba mal con el radio de Hough y propuso medir el diámetro con el área del contorno. También adaptó ese código al bucle de la cámara.
- **Tarea 2:** propuso el esquema de trabajo (segmentación, características de SMACC, clasificador de vecinos más cercanos y evaluación) y escribió el código del cuaderno.
- **README:** el borrador de este documento también se hizo con su ayuda, a partir de nuestro código y del README de la práctica anterior.

### 5. Bibliografía y Fuentes Consultadas

- [SMACC: A System for Microplastics Automatic Counting and Classification](https://doi.org/10.1109/ACCESS.2020.2970498)
- [OpenCV: mouse events (highgui)](https://docs.opencv.org/4.13.0/d7/dfc/group__highgui.html#ga89e7806b0a616f6f1d502bd8c183ad3e)
- [OpenCV: Feature Detection (HoughCircles)](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html)
- [OpenCV: Structural Analysis and Shape Descriptors](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
