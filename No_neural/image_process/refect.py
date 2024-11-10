import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cv2

# Leer la imagen
img = mimg.imread('image_process/lena.tif')
img_a = img.copy()

m, n, c = img_a.shape

#Funcion para leer la imagen con matplotlib
def leer_imagen(path):
    img = mimg.imread(path)
    return img

# Funcion para manipular el canal de color
def manipular_canal(img, canal, factor, avanzado=True):
    img_a = img.copy()
    # m, n, c = img_a.shape
    # for i in range(m):
    #     for j in range(n):
    #         img_a[i, j, canal] = img[i, j, canal] * factor
    
    if not avanzado:
        img_a[:,:,canal] = img[:,:,canal] * factor
    else:
        m, n, c = img_a.shape
        for i in range(m):
            for j in range(n):
                if img[i, j, canal] > 200:
                    img_a[i, j, canal] = img[i, j, canal] * factor
    return img_a

# Funcion binarizar imagen
def binarizar_imagen(img, canal, umbral):
    img_a = img.copy()
    m, n, c = img_a.shape
    for i in range(m):
        for j in range(n):
            pixel = img[i, j]
            if pixel[canal] > umbral:
                img_a[i, j] = 255, 255, 255
            else:
                img_a[i, j] = 0, 0, 0
    return img_a

# Funcion para modificar el brillo de la imagen
def brillo(img, factor):
    img_a = img.copy()
    img_a[:,:,:] = img[:,:,:] * factor
    return img_a

# Funcion para cambiar a escala de grises
def gris(img):
    img_a = img.copy()
    m, n, c = img_a.shape
    for i in range(m):
        for j in range(n):
            pixel = img[i, j]
            promedio = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) // 3
            img_a[i, j] = promedio, promedio, promedio
    return img_a

# Equalizar histograma
def equalizar_histograma(img):
    img_a = img.copy()
    for i in range(3):
        img_a[:,:,i] = cv2.equalizeHist(img_a[:,:,i])
    return img_a

# Equalizar histograma con CLAHE
def equalizar_CLAHE(img, clip, tile): #CLAHE: Contrast Limited Adaptive Histogram Equalization
    img_a = img.copy()
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    for i in range(3):
        img_a[:,:,i] = clahe.apply(img_a[:,:,i])
    return img_a

# Recortar imagen
def recortar_imagen(img, x1, y1, x2, y2):
    img_a = img.copy()
    img_a = img_a[y1:y2, x1:x2]
    return img_a

# Rotar imagen
def rotar_imagen(img, angulo):
    img_a = img.copy()
    m, n, c = img_a.shape
    centro = (n // 2, m // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1)
    img_a = cv2.warpAffine(img_a, M, (n, m))
    return img_a

def espejo(img, eje):
    img_a = img.copy()
    img_a = cv2.flip(img_a, eje)
    return img_a

# Funcion para mostrar la imagen y subplots
def mostrar_imagen(img, img_a, canal, selector):
    # Crear la figura con subplots
    plt.figure(figsize=(15, 15))

    # Primer subplot: Imagen original
    plt.subplot(2, 2, 1)  # 2 filas, 2 columnas, posición 1
    plt.imshow(img)
    plt.title('Imagen Original')
    plt.axis('off')  # Ocultar los ejes

    # Segundo subplot: Histograma de colores
    plt.subplot(2, 2, 2)  # 2 filas, 2 columnas, posición 2
    color = ('r', 'g', 'b')  # Colores para el histograma
    plt.title('Histograma de colores')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])  # Limitar el rango del eje X

    # Tercer subplot: Imagen en escala de grises
    plt.subplot(2, 2, 3)  # 2 filas, 1 columna, posición 3 (abajo)
    plt.imshow(img_a)
    plt.title(f'Imagen modificada canal: {canal} - {selector}')
    plt.axis('off')  # Ocultar los ejes

    # Cuarto subplot: Histograma de colores
    plt.subplot(2, 2, 4)  # 2 filas, 2 columnas, posición 4
    color = ('r', 'g', 'b')  # Colores para el histograma
    plt.title('Histograma de colores')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img_a], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])  # Limitar el rango del eje X

    # Mostrar la figura completa
    plt.tight_layout()
    plt.show()

# Funcion principal
def main():
    img = leer_imagen('image_process/lena.tif')
    canal = 'Algo'
    selector = 'Algo'
    img_a = manipular_canal(img, canal=0, factor=0.75, avanzado=False)
    img_a = equalizar_CLAHE(img_a, clip=2, tile=8)
    #img_a = recortar_imagen(img_a, 222, 200, 362, 390)
    #img_a = rotar_imagen(img_a, 90)
    img_a = espejo(img_a, 1)

    # Canal de color a manipular, forzar sleccion de canal 0,1,2
    # while True:
    #     try:
    #         canal = int(input('Ingrese el canal de color a manipular (0, 1, 2): '))
    #         if canal == 0 or canal == 1 or canal == 2:
    #             break
    #         raise ValueError
    #     except ValueError:
    #         print('Ingrese un valor válido')

    # while True:
    #     try:
    #         selector = int(input('Ingrese si se quiere binarizar (1) o manipular el canal (2) o manipular brillo (3) o cambiar a gris (4): '))
    #         if selector == 1 or selector == 2 or selector == 3 or selector == 4:
    #             break
    #         raise ValueError
    #     except ValueError:
    #         print('Ingrese un valor válido')
    
    # if selector == 1:
    #     selector = 'Binarizar'
    #     while True:
    #         try:
    #             umbral = int(input('Ingrese el umbral para binarizar la imagen (por ejemplo 127): '))
    #             if umbral < 0 or umbral > 255:
    #                 raise ValueError
    #             break
    #         except ValueError:
    #             print('Ingrese un valor válido')
    #     img_a = binarizar_imagen(img, canal=int(canal), umbral=umbral)
    # elif selector == 2:
    #     selector = 'Manipular'
    #     while True:
    #         try:
    #             factor = float(input('Ingrese el factor para manipular el canal de color (por ejemplo .5): '))
    #             if factor < 0 or factor > 1:
    #                 raise ValueError
    #             break
    #         except ValueError:
    #             print('Ingrese un valor válido')
    #     img_a = manipular_canal(img, canal=int(canal), factor=factor, avanzado=False)
    # elif selector == 3:
    #     selector = 'Brillo'
    #     while True:
    #         try:
    #             factor = float(input('Ingrese el factor para modificar el brillo de la imagen (por ejemplo .5): '))
    #             if factor < 0:
    #                 raise ValueError
    #             break
    #         except ValueError:
    #             print('Ingrese un valor válido')
    #     img_a = brillo(img, factor=factor)
    #     canal = 'todos'
    # elif selector == 4:
    #     selector = 'Gris'
    #     img_a = gris(img)
    #     canal = 'todos'
    
    # Mostrar la imagen
    mostrar_imagen(img, img_a, canal, selector)

if __name__ == '__main__':
    main()