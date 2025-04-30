def validar(board, row, col, num):
    # Itera sobre las 9 celdas de una fila, columna y subcuadro 3x3
    for i in range(9):
        # Verifica si el número ya está en la fila
        if board[row][i] == num or board[i][col] == num:
            return False
        # Verifica si el número ya está en el subcuadro 3x3 correspondiente
        if board[3*(row//3) + i//3][3*(col//3) + i%3] == num:
            return False
    # Si el número no está en la fila, columna ni subcuadro, es válido
    return True

def resolver(board):
    # Itera sobre todas las filas del tablero
    for row in range(9):
        # Itera sobre todas las columnas del tablero
        for col in range(9):
            # Si encuentra una celda vacía (valor 0)
            if board[row][col] == 0:
                # Prueba números del 1 al 9
                for num in range(1, 10):
                    # Verifica si el número es válido en la celda actual
                    if validar(board, row, col, num):
                        # Asigna el número a la celda
                        board[row][col] = num
                        # Llama recursivamente para resolver el resto del tablero
                        if resolver(board):
                            return True
                        # Si no se puede resolver, deshace el cambio (backtracking)
                        board[row][col] = 0
                # Si ningún número es válido, retorna False (no se puede resolver)
                return False
    # Si no hay celdas vacías, el tablero está resuelto
    return True
