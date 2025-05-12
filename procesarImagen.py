import cv2
import numpy as np
import tensorflow as tf

modeloPrint = tf.keras.models.load_model("modelo_sudoku.h5")

def mejorar_contraste_con_clahe(imagen_bgr):
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gris)

def extraer_digito_con_contraste(celda_bgr, area_minima=50):
    gris_clahe = mejorar_contraste_con_clahe(celda_bgr)
    borrosa = cv2.GaussianBlur(gris_clahe, (3, 3), 0)
    _, binaria = cv2.threshold(borrosa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaria = cv2.erode(binaria, np.ones((2, 2), np.uint8), iterations=1)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        if area > area_minima:
            x, y, w, h = cv2.boundingRect(contorno)
            recorte = binaria[y:y+h, x:x+w]
            tam_max = max(w, h)
            lienzo = np.zeros((tam_max, tam_max), dtype=np.uint8)
            x_offset = (tam_max - w) // 2
            y_offset = (tam_max - h) // 2
            lienzo[y_offset:y_offset+h, x_offset:x_offset+w] = recorte
            return cv2.resize(lienzo, (28, 28), interpolation=cv2.INTER_NEAREST)
    return np.zeros((28, 28), dtype=np.uint8)

def predecir_celda_con_modelo(celda, modelo):
    entrada = celda.astype("float32") / 255.0
    entrada = entrada.reshape(1, 28, 28, 1)
    salida = modelo.predict(entrada, verbose=0)
    return int(np.argmax(salida))

def encontrar_cuadro_sudoku(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    borrosa = cv2.GaussianBlur(gris, (9, 9), 0)
    _, umbral = cv2.threshold(borrosa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    invertida = cv2.bitwise_not(umbral)
    contornos, _ = cv2.findContours(invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aprox) == 4:
            return np.float32([p[0] for p in aprox])
    return None

def ordenar_puntos(puntos):
    puntos = puntos[np.argsort(puntos[:, 0])]
    izquierda = puntos[:2]
    derecha = puntos[2:]

    izquierda = izquierda[np.argsort(izquierda[:, 1])]
    derecha = derecha[np.argsort(derecha[:, 1])]

    return np.array([izquierda[0], derecha[0], derecha[1], izquierda[1]], dtype="float32")

def procesar_imagen_a_tablero(image_path):
    img = cv2.imread(image_path)
    original = img.copy()

    puntos = encontrar_cuadro_sudoku(img)
    if puntos is None:
        print("No se encontró el tablero de Sudoku.")
        return None

    puntos_ordenados = ordenar_puntos(puntos)
    destino = np.array([
        [0, 0],
        [449, 0],
        [449, 449],
        [0, 449]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(puntos_ordenados, destino)
    warp = cv2.warpPerspective(original, M, (450, 450))

    tam_celda = 450 // 9
    celdas_procesadas = []

    for fila in range(9):
        for col in range(9):
            x1, y1 = col * tam_celda, fila * tam_celda
            x2, y2 = (col + 1) * tam_celda, (fila + 1) * tam_celda
            celda = warp[y1:y2, x1:x2]
            limpia = extraer_digito_con_contraste(celda)
            celdas_procesadas.append(limpia)

    tablero = np.array([predecir_celda_con_modelo(c, modeloPrint) for c in celdas_procesadas]).reshape(9, 9)
    print("Tablero procesado:")
    print(tablero)
    return tablero