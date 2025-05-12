import cv2  # Importa OpenCV para procesamiento de imágenes
import numpy as np  # Importa NumPy para operaciones con matrices
import tensorflow as tf  # Importa TensorFlow para trabajar con modelos de aprendizaje profundo

# Carga el modelo preentrenado para reconocer dígitos en el Sudoku
modeloPrint = tf.keras.models.load_model("modelo_sudoku.h5")

# Mejora el contraste de una imagen utilizando el algoritmo CLAHE
def mejorar_contraste_con_clahe(imagen_bgr):
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Configura CLAHE
    return clahe.apply(gris)  # Aplica CLAHE y devuelve la imagen mejorada

# Extrae un dígito de una celda aplicando técnicas de procesamiento de imágenes
def extraer_digito_con_contraste(celda_bgr, area_minima=50):
    gris_clahe = mejorar_contraste_con_clahe(celda_bgr)  # Mejora el contraste de la celda
    borrosa = cv2.GaussianBlur(gris_clahe, (3, 3), 0)  # Aplica desenfoque gaussiano
    _, binaria = cv2.threshold(borrosa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binariza la imagen
    binaria = cv2.erode(binaria, np.ones((2, 2), np.uint8), iterations=1)  # Erosiona para limpiar ruido

    # Encuentra contornos en la imagen binarizada
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:  # Si se encuentran contornos
        contorno = max(contornos, key=cv2.contourArea)  # Selecciona el contorno más grande
        area = cv2.contourArea(contorno)  # Calcula el área del contorno
        if area > area_minima:  # Si el área es mayor al mínimo
            x, y, w, h = cv2.boundingRect(contorno)  # Obtiene el rectángulo delimitador
            recorte = binaria[y:y+h, x:x+w]  # Recorta la región del contorno
            tam_max = max(w, h)  # Determina el tamaño máximo entre ancho y alto
            lienzo = np.zeros((tam_max, tam_max), dtype=np.uint8)  # Crea un lienzo cuadrado negro
            x_offset = (tam_max - w) // 2  # Calcula el desplazamiento horizontal
            y_offset = (tam_max - h) // 2  # Calcula el desplazamiento vertical
            lienzo[y_offset:y_offset+h, x_offset:x_offset+w] = recorte  # Coloca el recorte en el lienzo
            return cv2.resize(lienzo, (28, 28), interpolation=cv2.INTER_NEAREST)  # Redimensiona a 28x28
    return np.zeros((28, 28), dtype=np.uint8)  # Devuelve una imagen vacía si no hay dígito

# Predice el número en una celda utilizando el modelo cargado
def predecir_celda_con_modelo(celda, modelo):
    entrada = celda.astype("float32") / 255.0  # Normaliza los valores de la celda
    entrada = entrada.reshape(1, 28, 28, 1)  # Ajusta la forma para el modelo
    salida = modelo.predict(entrada, verbose=0)  # Realiza la predicción
    return int(np.argmax(salida))  # Devuelve el número con mayor probabilidad

# Encuentra el contorno del cuadro del Sudoku en la imagen
def encontrar_cuadro_sudoku(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
    borrosa = cv2.GaussianBlur(gris, (9, 9), 0)  # Aplica desenfoque gaussiano
    _, umbral = cv2.threshold(borrosa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binariza la imagen
    invertida = cv2.bitwise_not(umbral)  # Invierte los colores de la imagen
    contornos, _ = cv2.findContours(invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encuentra contornos
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)  # Ordena los contornos por área

    for contorno in contornos:  # Itera sobre los contornos
        perimetro = cv2.arcLength(contorno, True)  # Calcula el perímetro del contorno
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)  # Aproxima el contorno a un polígono
        if len(aprox) == 4:  # Si el polígono tiene 4 lados
            return np.float32([p[0] for p in aprox])  # Devuelve los puntos del contorno
    return None  # Devuelve None si no se encuentra un cuadro

# Ordena los puntos de un contorno en el orden correcto
def ordenar_puntos(puntos):
    puntos = puntos[np.argsort(puntos[:, 0])]  # Ordena los puntos por coordenada x
    izquierda = puntos[:2]  # Toma los dos puntos más a la izquierda
    derecha = puntos[2:]  # Toma los dos puntos más a la derecha

    izquierda = izquierda[np.argsort(izquierda[:, 1])]  # Ordena los puntos izquierdos por y
    derecha = derecha[np.argsort(derecha[:, 1])]  # Ordena los puntos derechos por y

    return np.array([izquierda[0], derecha[0], derecha[1], izquierda[1]], dtype="float32")  # Devuelve los puntos ordenados

# Procesa una imagen para extraer el tablero de Sudoku como una matriz
def procesar_imagen_a_tablero(image_path):
    img = cv2.imread(image_path)  # Carga la imagen desde el archivo
    original = img.copy()  # Crea una copia de la imagen original

    puntos = encontrar_cuadro_sudoku(img)  # Encuentra el cuadro del Sudoku
    if puntos is None:  # Si no se encuentra el cuadro
        print("No se encontró el tablero de Sudoku.")  # Muestra un mensaje de error
        return None  # Devuelve None

    puntos_ordenados = ordenar_puntos(puntos)  # Ordena los puntos del cuadro
    destino = np.array([  # Define los puntos destino para la transformación
        [0, 0],
        [449, 0],
        [449, 449],
        [0, 449]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(puntos_ordenados, destino)  # Calcula la transformación de perspectiva
    warp = cv2.warpPerspective(original, M, (450, 450))  # Aplica la transformación a la imagen

    tam_celda = 450 // 9  # Calcula el tamaño de cada celda
    celdas_procesadas = []  # Lista para almacenar las celdas procesadas

    for fila in range(9):  # Itera sobre las filas del tablero
        for col in range(9):  # Itera sobre las columnas del tablero
            x1, y1 = col * tam_celda, fila * tam_celda  # Calcula las coordenadas de la celda
            x2, y2 = (col + 1) * tam_celda, (fila + 1) * tam_celda
            celda = warp[y1:y2, x1:x2]  # Extrae la celda de la imagen
            limpia = extraer_digito_con_contraste(celda)  # Procesa la celda para extraer el dígito
            celdas_procesadas.append(limpia)  # Añade la celda procesada a la lista

    # Convierte las celdas procesadas en un tablero de 9x9 utilizando el modelo
    tablero = np.array([predecir_celda_con_modelo(c, modeloPrint) for c in celdas_procesadas]).reshape(9, 9)
    print("Tablero procesado:")  # Muestra el tablero procesado
    print(tablero)
    return tablero  # Devuelve el tablero como una matriz