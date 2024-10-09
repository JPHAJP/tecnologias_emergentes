import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import cv2

# Leer la imagen
img = mimg.imread('image_process/lena.tif')
img_a = img.copy()

m, n, c = img_a.shape

#Funcion para leer la imagen con matplotlib
def leer_imagen(path):
    img = mimg.imread(path)
    return img

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
    plt.savefig('image_process/lena_modificada.png')
    #plt.show()

def filtro(img, kernel):
    m, n, c = img.shape
    img_a = img.copy()
    for i in range(1,m-1):
        for j in range(1,n-1):
            for k in range(c):
                suma = \
                img_a[i-1,j-1,k]*kernel[0,0] + img_a[i-1,j,k]*kernel[0,1] + img_a[i-1,j+1,k]*kernel[0,2] + \
                img_a[i,j-1,k]*kernel[1,0] + img_a[i,j,k]*kernel[1,1] + img_a[i,j+1,k]*kernel[1,2] + \
                img_a[i+1,j-1,k]*kernel[2,0] + img_a[i+1,j,k]*kernel[2,1] + img_a[i+1,j+1,k]*kernel[2,2]
            
    img_a = cv2.filter2D(img_a, -1, kernel)
    return img_a

def filtro_cv2(img, kernel):
    img_a = cv2.filter2D(img, -1, kernel)
    return img_a

def kernel_gaussiano():
    kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])/9
    kernel_2 = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])/16
    kernel_3 = np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]])/412
    kernel_4 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    

    return kernel_4


def main():
    img = leer_imagen('image_process/lena.tif')
    #Filtrar imagen 
    kernel = kernel_gaussiano()
    #img_a = filtro(img, kernel) #Lento
    img_a = filtro_cv2(img, kernel)
    mostrar_imagen(img, img_a, canal='N/A', selector='Filtro')

if __name__ == '__main__':
    main()

# Tarea hacer un kernel que detecte los bordes verticales, horizontales y todos los bordes en blanco y negro.
# Plotear img original y las 3 modificadas
