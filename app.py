from flask import Flask, render_template, request
from sudoku import resolver

app = Flask(__name__)

def preprocesar_imagen(path):
    pass

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
        return render_template('result.html', original=original_board, solved=board)
    return render_template('index.html')