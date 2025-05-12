from flask import Flask, render_template, request
from sudoku import resolver
from procesarImagen import procesar_imagen_a_tablero
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        path = 'static/uploaded_sudoku.jpg'
        img_file.save(path)

        # Tablero de ejemplo
        # board = np.array([
        #     [5, 3, 0, 0, 7, 0, 0, 0, 0],
        #     [6, 0, 0, 1, 9, 5, 0, 0, 0],
        #     [0, 9, 8, 0, 0, 0, 0, 6, 0],
        #     [8, 0, 0, 0, 6, 0, 0, 0, 3],
        #     [4, 0, 0, 8, 0, 3, 0, 0, 1],
        #     [7, 0, 0, 0, 2, 0, 0, 0, 6],
        #     [0, 6, 0, 0, 0, 0, 2, 8, 0],
        #     [0, 0, 0, 4, 1, 9, 0, 0, 5],
        #     [0, 0, 0, 0, 8, 0, 0, 7, 9]
        # ])

        board = procesar_imagen_a_tablero(path)  # Procesar imagen a tablero
        original_board = [row.tolist() for row in board]  # Convertir para renderizar

        resolver(board)
        return render_template('result.html', original=original_board, resuelto=board.tolist())
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
