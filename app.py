from flask import Flask, render_template, request
from sudoku import resolver
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('modelo.h5')

def ordenar_puntos(puntos):
    puntos = puntos.reshape((4, 2))
    suma = puntos.sum(axis=1)
    diff = np.diff(puntos, axis=1)

    ordenados = np.zeros((4, 2), dtype="float32")
    ordenados[0] = puntos[np.argmin(suma)]  # top-left
    ordenados[2] = puntos[np.argmax(suma)]  # bottom-right
    ordenados[1] = puntos[np.argmin(diff)]  # top-right
    ordenados[3] = puntos[np.argmax(diff)]  # bottom-left
    return ordenados

def encontrar_tablero(img):
    contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mayor_area = 0
    mejor_contorno = None

    for contorno in contornos:
        peri = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
        if len(aprox) == 4 and cv2.contourArea(aprox) > mayor_area:
            mayor_area = cv2.contourArea(aprox)
            mejor_contorno = aprox

    return mejor_contorno

def preprocesar_imagen(path):
    img_original = cv2.imread(path)
    img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_filtrada = cv2.GaussianBlur(img_gris, (5, 5), 0)
    _, img_binaria = cv2.threshold(img_filtrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contorno_tablero = encontrar_tablero(img_binaria)
    if contorno_tablero is None:
        raise Exception("No se pudo encontrar el tablero de Sudoku")

    puntos = ordenar_puntos(contorno_tablero)
    lado = max(
        int(np.linalg.norm(puntos[0] - puntos[1])),
        int(np.linalg.norm(puntos[1] - puntos[2])),
        int(np.linalg.norm(puntos[2] - puntos[3])),
        int(np.linalg.norm(puntos[3] - puntos[0]))
    )

    destino = np.array([[0, 0], [lado - 1, 0], [lado - 1, lado - 1], [0, lado - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(puntos, destino)
    img_transformada = cv2.warpPerspective(img_gris, M, (lado, lado))

    celdas = []
    paso = lado // 9

    for i in range(9):
        fila = []
        for j in range(9):
            x, y = j * paso, i * paso
            celda = img_transformada[y:y+paso, x:x+paso]
            celda = cv2.resize(celda, (28, 28))
            celda_bin = cv2.adaptiveThreshold(celda, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            if np.sum(celda_bin) < 50:
                fila.append(0)
                continue

            entrada = celda_bin / 255.0
            entrada = entrada.reshape(1, 28, 28, 1)
            pred = model.predict(entrada)
            digit = int(np.argmax(pred)) if pred.max() > 0.7 else 0
            print(f"Predicción para celda ({i}, {j}): {digit}")

            fila.append(digit)
        celdas.append(fila)

    return celdas

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        path = 'static/uploaded_sudoku.jpg'
        img_file.save(path)

        board = preprocesar_imagen(path)
        original_board = [row[:] for row in board]

        resolver(board)
        return render_template('result.html', original=original_board, resuelto=board)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
