#Codigo para crear matrices segun el input del usuario y multiplicarlas con producto punto

import random

def crear_matriz(filas, columnas):
    matriz = []
    if input("Desea ingresar los valores de la matriz? (s/n): ") == "s":
        for i in range(filas):
            matriz.append([])
            for j in range(columnas):
                matriz[i].append(int(input(f"Ingrese el valor de la fila {i + 1}, columna {j + 1}: ")))
        return matriz
    else:
        for i in range(filas):
            matriz.append([])
            for j in range(columnas):
                matriz[i].append(random.randint(1, 10))
        return matriz

def producto_punto(matriz1, matriz2):
    if len(matriz1[0]) != len(matriz2):
        return None
    matriz_resultado = []
    for i in range(len(matriz1)):
        matriz_resultado.append([])
        for j in range(len(matriz2[0])):
            suma = 0
            for k in range(len(matriz1[0])):
                suma += matriz1[i][k] * matriz2[k][j]
            matriz_resultado[i].append(suma)
    return matriz_resultado

def input_matriz():
    filas = int(input("Ingrese el numero de filas de la matriz: "))
    columnas = int(input("Ingrese el numero de columnas de la matriz: "))
    matriz = crear_matriz(filas, columnas)
    print("Matriz:")
    for fila in matriz:
        print(fila)
    return matriz

def main():
    print("Multiplicacion de matrices")
    matriz1 = input_matriz()
    matriz2 = input_matriz()

    matriz_resultado = producto_punto(matriz1, matriz2)
    if matriz_resultado is None:
        print("No se pueden multiplicar las matrices")
    else:
        print("Matriz resultado:")
        for fila in matriz_resultado:
            print(fila)

if __name__ == "__main__":
    main()