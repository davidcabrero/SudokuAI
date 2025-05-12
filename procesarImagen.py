import cv2
import numpy as np
import tensorflow as tf
import os

modeloPrint = tf.keras.models.load_model("modelo_sudoku.h5")

# Función para mejorar el contraste de una imagen usando CLAHE (Contrast Limited Adaptive Histogram Equalization)
def mejorar_contraste_con_clahe(imagen_bgr):
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Crear un objeto CLAHE con parámetros específicos
    return clahe.apply(gris)  # Aplicar CLAHE a la imagen en escala de grises

# Función para extraer un dígito de una celda de Sudoku mejorando el contraste
def extraer_digito_con_contraste(celda_bgr, area_minima=50):
    gris_clahe = mejorar_contraste_con_clahe(celda_bgr)  # Mejorar el contraste de la celda
    borrosa = cv2.GaussianBlur(gris_clahe, (3, 3), 0)  # Aplicar desenfoque gaussiano para suavizar la imagen
    _, binaria = cv2.threshold(borrosa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binarizar la imagen
    binaria = cv2.erode(binaria, np.ones((2, 2), np.uint8), iterations=1)  # Erosionar para eliminar ruido

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontrar contornos
    if contornos:  # Si se encontraron contornos
        contorno = max(contornos, key=cv2.contourArea)  # Seleccionar el contorno con mayor área
        area = cv2.contourArea(contorno)  # Calcular el área del contorno
        if area > area_minima:  # Si el área es mayor al mínimo especificado
            x, y, w, h = cv2.boundingRect(contorno)  # Obtener el rectángulo delimitador del contorno
            recorte = binaria[y:y+h, x:x+w]  # Recortar la región del contorno
            tam_max = max(w, h)  # Determinar el tamaño máximo entre ancho y alto
            lienzo = np.zeros((tam_max, tam_max), dtype=np.uint8)  # Crear un lienzo cuadrado negro
            x_offset = (tam_max - w) // 2  # Calcular el desplazamiento horizontal
            y_offset = (tam_max - h) // 2  # Calcular el desplazamiento vertical
            lienzo[y_offset:y_offset+h, x_offset:x_offset+w] = recorte  # Colocar el recorte en el lienzo
            return cv2.resize(lienzo, (28, 28), interpolation=cv2.INTER_NEAREST)  # Redimensionar a 28x28
    return np.zeros((28, 28), dtype=np.uint8)  # Devolver una imagen vacía si no se encuentra un dígito

# Función para predecir el valor de una celda usando un modelo de TensorFlow
def predecir_celda_con_modelo(celda, modelo):
    entrada = celda.astype("float32") / 255.0  # Normalizar los valores de la celda entre 0 y 1
    entrada = entrada.reshape(1, 28, 28, 1)  # Ajustar la forma de la entrada para el modelo
    salida = modelo.predict(entrada, verbose=0)  # Realizar la predicción con el modelo
    return int(np.argmax(salida))  # Devolver el índice de la clase con mayor probabilidad

# Función para procesar una imagen y convertirla en un tablero de Sudoku
def procesar_imagen_a_tablero(image_path):
    img = cv2.imread(image_path)  # Leer la imagen desde la ruta especificada
    img_original = img.copy()  # Crear una copia de la imagen original
    alto, ancho = img.shape[:2]  # Obtener las dimensiones de la imagen

    # Definir los puntos de las esquinas de la imagen original
    pts_ordenados = np.array([
        [0, 0],
        [ancho - 1, 0],
        [ancho - 1, alto - 1],
        [0, alto - 1]
    ], dtype="float32")

    # Definir los puntos de destino para transformar la perspectiva
    destino = np.array([
        [0, 0],
        [449, 0],
        [449, 449],
        [0, 449]
    ], dtype="float32")

    # Calcular la matriz de transformación de perspectiva
    M = cv2.getPerspectiveTransform(pts_ordenados, destino)
    # Aplicar la transformación de perspectiva
    warp = cv2.warpPerspective(img_original, M, (450, 450), flags=cv2.INTER_NEAREST)

    # Calcular el tamaño de cada celda del Sudoku
    tam_celda_x = 450 // 9
    tam_celda_y = 450 // 9

    celdas_procesadas = []  # Lista para almacenar las celdas procesadas

    # Iterar sobre cada celda del Sudoku
    for fila in range(9):
        for col in range(9):
            x1 = col * tam_celda_x  # Coordenada x inicial de la celda
            y1 = fila * tam_celda_y  # Coordenada y inicial de la celda
            x2 = (col + 1) * tam_celda_x  # Coordenada x final de la celda
            y2 = (fila + 1) * tam_celda_y  # Coordenada y final de la celda
            celda = warp[y1:y2, x1:x2]  # Extraer la celda de la imagen transformada
            limpia = extraer_digito_con_contraste(celda)  # Procesar la celda para extraer el dígito
            celdas_procesadas.append(limpia)  # Agregar la celda procesada a la lista

    # Predecir los valores de todas las celdas y convertirlos en un tablero 9x9
    tablero = np.array([predecir_celda_con_modelo(c, modeloPrint) for c in celdas_procesadas]).reshape(9, 9)
    print("Tablero procesado:")
    print(tablero)
    return tablero
