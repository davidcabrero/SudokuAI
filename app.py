from flask import Flask, render_template, request
from sudoku import resolver
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('modelo.h5')

def ordenar_puntos(puntos):
    # Reorganizar los puntos para que estén en un formato adecuado
    puntos = puntos.reshape((4, 2))  # Asegurarse de que los puntos estén en una matriz de 4x2
    suma = puntos.sum(axis=1)  # Sumar las coordenadas x e y de cada punto
    diff = np.diff(puntos, axis=1)  # Calcular la diferencia entre las coordenadas x e y de cada punto

    # Crear una matriz para almacenar los puntos ordenados
    ordenados = np.zeros((4, 2), dtype="float32")  # Inicializar una matriz de 4x2 con ceros
    ordenados[0] = puntos[np.argmin(suma)]  # top-left
    ordenados[2] = puntos[np.argmax(suma)]  # bottom-right
    ordenados[1] = puntos[np.argmin(diff)]  # top-right
    ordenados[3] = puntos[np.argmax(diff)]  # bottom-left
    return ordenados

def encontrar_tablero(img):
    # Encontrar todos los contornos en la imagen binaria
    contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializar variables para almacenar el contorno con el área más grande
    mayor_area = 0
    mejor_contorno = None

    # Iterar sobre cada contorno encontrado
    for contorno in contornos:
        # Calcular el perímetro del contorno
        peri = cv2.arcLength(contorno, True)
        
        # Aproximar el contorno a un polígono con menos vértices
        aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
        
        # Verificar si el contorno tiene 4 lados (posible cuadrado) y si su área es mayor que la mayor área encontrada hasta ahora
        if len(aprox) == 4 and cv2.contourArea(aprox) > mayor_area:
            # Actualizar la mayor área y el mejor contorno
            mayor_area = cv2.contourArea(aprox)
            mejor_contorno = aprox

    return mejor_contorno

def preprocesar_imagen(path):
    # Leer la imagen desde la ruta proporcionada
    img_original = cv2.imread(path)
    
    # Convertir la imagen a escala de grises
    img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro Gaussiano para suavizar la imagen
    img_filtrada = cv2.GaussianBlur(img_gris, (5, 5), 0)
    
    # Aplicar un umbral adaptativo para binarizar la imagen
    _, img_binaria = cv2.threshold(img_filtrada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar el contorno del tablero de Sudoku
    contorno_tablero = encontrar_tablero(img_binaria)
    if contorno_tablero is None:
        # Lanzar una excepción si no se encuentra el tablero
        raise Exception("No se pudo encontrar el tablero de Sudoku")

    # Ordenar los puntos del contorno para facilitar la transformación
    puntos = ordenar_puntos(contorno_tablero)
    
    # Calcular el tamaño del lado del tablero
    lado = max(
        int(np.linalg.norm(puntos[0] - puntos[1])),  # Distancia entre el punto superior izquierdo y derecho
        int(np.linalg.norm(puntos[1] - puntos[2])),  # Distancia entre el punto superior derecho e inferior derecho
        int(np.linalg.norm(puntos[2] - puntos[3])),  # Distancia entre el punto inferior derecho e inferior izquierdo
        int(np.linalg.norm(puntos[3] - puntos[0]))   # Distancia entre el punto inferior izquierdo y superior izquierdo
    )

    # Definir los puntos de destino para la transformación de perspectiva
    destino = np.array([[0, 0], [lado - 1, 0], [lado - 1, lado - 1], [0, lado - 1]], dtype="float32")
    
    # Calcular la matriz de transformación de perspectiva
    M = cv2.getPerspectiveTransform(puntos, destino)
    
    # Aplicar la transformación de perspectiva para obtener una vista rectificada del tablero
    img_transformada = cv2.warpPerspective(img_gris, M, (lado, lado))

    # Dividir la imagen transformada en celdas individuales
    celdas = []
    paso = lado // 9  # Tamaño de cada celda (asumiendo un tablero de 9x9)

    for i in range(9):
        fila = []
        for j in range(9):
            # Extraer cada celda de la imagen transformada
            x, y = j * paso, i * paso
            celda = img_transformada[y:y+paso, x:x+paso]
            
            # Redimensionar la celda a 28x28 píxeles
            celda = cv2.resize(celda, (28, 28))
            
            # Aplicar un umbral adaptativo para binarizar la celda
            celda_bin = cv2.adaptiveThreshold(celda, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Verificar si la celda está vacía (poca suma de píxeles blancos)
            if np.sum(celda_bin) < 50:
                fila.append(0)  # Agregar un 0 si la celda está vacía
                continue

            # Preparar la celda para la predicción del modelo
            entrada = celda_bin / 255.0  # Normalizar los valores de píxeles
            entrada = entrada.reshape(1, 28, 28, 1)  # Ajustar la forma para el modelo

            # Realizar la predicción del dígito en la celda
            pred = model.predict(entrada)
            digit = int(np.argmax(pred)) if pred.max() > 0.7 else 0  # Asignar 0 si la confianza es baja
            
            # Imprimir la predicción para depuración
            print(f"Predicción para celda ({i}, {j}): {digit}")

            # Agregar el dígito predicho a la fila
            fila.append(digit)
        # Agregar la fila al tablero
        celdas.append(fila)

    # Devolver el tablero procesado
    return celdas

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Imagen subida
        img_file = request.files['image']
        path = 'static/uploaded_sudoku.jpg'
        img_file.save(path)

        board = preprocesar_imagen(path) # Tablero
        original_board = [row[:] for row in board] # Copia del tablero original

        resolver(board) # Resolver el Sudoku
        return render_template('result.html', original=original_board, resuelto=board)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
