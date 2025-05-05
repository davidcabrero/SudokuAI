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
    img = cv2.resize(img, (252, 252))
    
    # Aplicar un umbral adaptativo combinado con Otsu
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
    
    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kernel más grande
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cells = []

    for i in range(9):
        row = []
        for j in range(9):
            # Extraer cada celda
            cell = thresh[i*28:(i+1)*28, j*28:(j+1)*28]
            
            # Redimensionar la celda al tamaño esperado por el modelo
            cell = cv2.resize(cell, (28, 28))
            
            # Normalizar los valores de píxeles
            cell = cell / 255.0
            
            # Añadir una dimensión para que sea compatible con el modelo
            cell = cell.reshape(1, 28, 28, 1)

            # Realizar la predicción
            prediction = model.predict(cell)
            
            # Ajustar el umbral de confianza dinámicamente
            confidence_threshold = max(0.7, prediction.max() - 0.1)
            digit = int(np.argmax(prediction)) if prediction.max() > confidence_threshold else 0
            
            # Depuración: Imprimir predicciones con confianza
            print(f"Predicción para celda ({i}, {j}): {prediction}, Digit: {digit}")
            
            # Validar predicción: Si es dudosa, intentar preprocesar de nuevo
            if digit == 0 and prediction.max() < confidence_threshold:
                cell = cv2.dilate(cell[0], kernel, iterations=1)  # Dilatar para mejorar bordes
                cell = cell.reshape(1, 28, 28, 1)
                prediction = model.predict(cell)
                digit = int(np.argmax(prediction)) if prediction.max() > confidence_threshold else 0
            
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

        board = preprocesar_imagen(path) # Procesa la imagen
        original_board = [row[:] for row in board] 

        resolver(board)
        return render_template('result.html', original=original_board, resuelto=board)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)