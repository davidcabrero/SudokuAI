# SudokuAI

SudokuAI es un proyecto diseñado para resolver sudokus automáticamente utilizando un algoritmo y redes neuronales.

## Características

- **Reconocimiento de Tableros**: Detecta y extrae el tablero de sudoku desde una imagen.
- **Resolución Automática**: Resuelve el sudoku utilizando algoritmos.

## Instalación

1. Clona este repositorio:
     ```bash
     git clone https://github.com/davidcabrero/SudokuAI.git
     cd SudokuAI
     ```
2. Instala las dependencias:
     ```bash
     pip install -r requirements.txt
     ```

## Uso

1. Ejecuta el script principal:
     ```bash
     python app.py
     ```
2. Sube la imagen de tu sudoku.
3. Muestra la resolución del sudoku.

## Estructura del Proyecto

- `app.py`: Programa principal.
- `procesarImagen.py`: Preprocesamiento de Imágenes.
- `sudoku.py`: Algoritmo de resolución del sudoku.
- `assets/`: Conjunto de datos para entrenar el modelo.
- `templates/`: Páginas de inicio y resultados.
- `modelo_sudoku.h5`: Modelo entrenado con Redes Neuronales CNN para el reconocimiento de números.
- `entrenar_modelo.ipynb`: Entrenamiento del modelo.
- `tests/`: Imágenes para realizar pruebas.