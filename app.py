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

        board = procesar_imagen_a_tablero(path)  # Procesar imagen a tablero
        original_board = [row.tolist() for row in board]  # Convertir para renderizar

        resolver(board)
        return render_template('result.html', original=original_board, resuelto=board.tolist())
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
