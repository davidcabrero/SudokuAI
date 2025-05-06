from flask import Flask, render_template, request
from sudoku import resolver
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('sudoku_model.h5') 

def preprocesar_imagen(path):
    """
    Preprocesa la imagen del Sudoku para extraer los dígitos en un formato de tablero.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Aumentar el contraste
    img = cv2.equalizeHist(img)
    
    # Redimensionar la imagen al tamaño esperado
    img = cv2.resize(img, (252, 252), interpolation=cv2.INTER_CUBIC)

    # Aplicar un filtro de mediana para reducir el ruido
    img = cv2.medianBlur(img, 3)

    # Aplicar un filtro de Gaussian para suavizar la imagen
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Aplicar un filtro de Sobel para detectar bordes
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y) 
    sobel = cv2.convertScaleAbs(sobel)  # Convertir a tipo de dato adecuado
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)  # Normalizar la imagen

    # aumentar el contorno del numero y agrandar el numero
    sobel = cv2.GaussianBlur(sobel, (5, 5), 0)  # Suavizar la imagen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sobel = cv2.dilate(sobel, kernel, iterations=1)  # Dilatar la imagen para aumentar el contorno
    
    # Invierte los colores, separa fondo y objetos
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
    
    # Operaciones morfológicas para eliminar ruido y mejorar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cells = []

    for i in range(9):
        row = []
        for j in range(9):
            # Extraer cada celda
            cell = thresh[i*28:(i+1)*28, j*28:(j+1)*28]
            
            # Redimensionar la celda
            cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_CUBIC)
            
            # Verificar si la celda está en blanco
            if np.sum(cell) < 50:  # Umbral bajo para detectar celdas vacías
                row.append(0)
                continue

            # Normalizar valores
            cell = cell / 255.0
            
            # Añadir dimensiones para que sea compatible con el modelo
            cell = cell.reshape(1, 28, 28, 1)

            # Dibujar contorno del numero
            cell_uint8 = (cell[0] * 255).astype(np.uint8)  # Convertir a tipo CV_8UC1
            contours, _ = cv2.findContours(cell_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(cell_uint8, [cnt], -1, (255, 255, 255), 1)

            # Realizar la predicción
            prediction = model.predict(cell)
            
            # Ajustar la confianza
            confidence_threshold = max(0.8, prediction.max() - 0.05)
            digit = int(np.argmax(prediction)) if prediction.max() > confidence_threshold else 0
            
            # Validar predicción
            if digit == 0 or prediction.max() < 0.60:  # Si el modelo predice un 0 o la confianza es baja
                    cell = cv2.dilate(cell[0], kernel, iterations=1)  # Dilatar la celda para mejorar la detección
                    cell = cell.reshape(1, 28, 28, 1)  # Reajustar dimensiones
                    prediction = model.predict(cell)  # Repetir predicción
                    digit = int(np.argmax(prediction)) if prediction.max() > confidence_threshold else 0  # Reajustar el dígito    

            # Depuración: Imprimir predicciones
            print(f"Predicción para celda ({i}, {j}): {prediction}, Digit: {digit}")
            
            row.append(digit)
        cells.append(row)
    return cells
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Sube la imagen del Sudoku
        img_file = request.files['image']
        path = 'static/uploaded_sudoku.jpg'
        img_file.save(path)

        # Tablero de sudoku de ejemplo para probar algoritmo
        # board = [
        #     [0, 0, 0, 2, 6, 0, 7, 0, 1],
        #     [6, 8, 0, 0, 7, 0, 0, 9, 0],
        #     [1, 9, 0, 0, 0, 4, 5, 0, 0],
        #     [8, 2, 0, 1, 0, 0, 0, 4, 0],
        #     [0, 0, 4, 6, 0, 2, 9, 0, 0],
        #     [0, 5, 0, 0, 0, 3, 0, 2, 8],
        #     [0, 0, 9, 3, 0, 0, 0, 7, 4],
        #     [0, 4, 0, 0, 5, 0, 0, 3, 6],
        #     [7, 0, 3, 0, 1, 8, 0, 0, 0]
        # ]

        board = preprocesar_imagen(path) # Procesa la imagen
        original_board = [row[:] for row in board] 

        resolver(board)
        return render_template('result.html', original=original_board, resuelto=board)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)