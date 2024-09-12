x = [1,0,0.0,1,
    0,1,0,1,0,
    0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1]

o = [0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1,
    0,1,0,1,0,
    0,0,1,0,0]

x_mal = [0,0,0.0,0,
        0,1,0,1,0,
        0,0,1,0,0,
        0,1,0,1,0,
        0,0,0,0,0]

moneda =   [1,0,1,0,0,
            0,1,0,1,0,
            1,0,1,0,0,
            0,1,0,1,0,
            0,0,1,0,1]

pesos =    [1,0,0,0,1,
            0,-1,0,-1,0,
            0,0,1,0,0,
            0,-1,0,-1,0,
            1,0,0,0,1]

bias = 1

def sumatoria(datos, pesos):
    suma = 0
    for i in range(len(datos)):
        suma += datos[i] * pesos[i]
    print(f'Suma: {suma}')
    return suma

def activacion(suma):
    return 1 if suma >= 0 else 0

    
def perceptron(datos, pesos, bias):
    #return activacion(sumatoria(datos, pesos) + bias)
    return 'X' if activacion(sumatoria(datos, pesos) + bias) == 1 else 'O'

print(f'Prediccion X: {perceptron(x, pesos, bias)}')
print(f'Prediccion 0: {perceptron(o, pesos, bias)}')
print(f'Prediccion Moneda: {perceptron(moneda, pesos, bias)}')
print(f'Prediccion X Mal: {perceptron(x_mal, pesos, bias)}')