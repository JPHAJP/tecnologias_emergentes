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
                        [1, 2, 1]])/9
    kernel_3 = np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]])/25
    kernel_4 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]) #detecta mas o menos los bordes
    return kernel_4

def combinar_bordes(img, kernel_x, kernel_y):
    # Convierte la imagen a escala de grises si aún no lo está
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Aplica los filtros
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)  # Usa CV_64F para evitar problemas de tipo
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    
    # Calcula la magnitud del gradiente
    grad_total = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    # Normaliza la imagen para que esté en el rango de 0 a 255
    grad_total = np.clip(grad_total, 0, 255)
    grad_total = grad_total.astype(np.uint8)  # Asegura que los valores sean enteros de 8 bits
    
    return grad_total


def kernel_bordes():
    kernel_vertical = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
    kernel_horizontal = np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]])
    return kernel_vertical, kernel_horizontal

def max_pooling(img, kernel_size):
    #img = umbralizar_imagen(img, 127)
    m, n = img.shape
    m_new = m // kernel_size
    n_new = n // kernel_size
    print(m,n,' New: ', m_new, n_new)
    img_a = np.zeros((m_new, n_new))
    for i in range(m_new):
        for j in range(n_new):
            img_a[i, j] = np.max(img[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size])

    #img_a=umbralizar_imagen(img_a, 127)
    return img_a

def umbralizar_imagen(img, umbral):
    img_a = img.copy()
    img_a[img_a < umbral] = 0
    img_a[img_a >= umbral] = 255
    return img_a

def main():
    img = leer_imagen('image_process/cyberpunk_citty.webp')
    #img = leer_imagen('image_process/lena.tif')
    #img = leer_imagen('image_process/JP.jpg')
    #img = leer_imagen('image_process/old_pic.jpg')
    kernel_vertical, kernel_horizontal = kernel_bordes()    

    #Convertir imagen a escala de grises
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Filtrar imagen con los diferentes kernels
    img_vertical = filtro_cv2(img, kernel_vertical)
    img_horizontal = filtro_cv2(img, kernel_horizontal)
    img_combined = max_pooling(combinar_bordes(img, kernel_vertical, kernel_horizontal),4)

    # Mostrar las imágenes resultantes
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray') 
    plt.title('Imagen Original')
    #plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_vertical, cmap='gray')
    plt.title('Bordes Verticales')
    #plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_horizontal, cmap='gray')
    plt.title('Bordes Horizontales')
    #plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(img_combined, cmap='gray')
    plt.title('Todos los Bordes')
    #plt.axis('off')

    plt.tight_layout()
    plt.show()


# def main():
#     img = leer_imagen('image_process/lena.tif')
#     #Filtrar imagen 
#     kernel = kernel_gaussiano()
#     #img_a = filtro(img, kernel) #Lento
#     img_a = filtro_cv2(img, kernel)
#     mostrar_imagen(img, img_a, canal='N/A', selector='Filtro')

if __name__ == '__main__':
    main()

# Los kernels que detectan bordes funcionan gracias a la idea de resaltar cambios rápidos en la intensidad de píxeles de una imagen, 
# lo cual suele ocurrir en los bordes de los objetos dentro de la imagen. Aquí te explico cada uno de ellos y la lógica detrás:

# ### 1. **Detección de bordes verticales** (Kernel Sobel en dirección X):
#    ```python
#    kernel_vertical = np.array([[-1, 0, 1],
#                                [-2, 0, 2],
#                                [-1, 0, 1]])
#    ```
#    - Este kernel resalta los cambios de intensidad en la dirección horizontal, lo que significa que detecta los bordes verticales.
#    - Los valores negativos a la izquierda del kernel y los positivos a la derecha indican que busca diferencias grandes de intensidad en esa dirección.
#    - Cuando aplicamos este kernel, cada píxel se multiplica por su valor correspondiente en la matriz, lo que da como resultado una imagen que destaca 
#       las áreas donde hay un cambio brusco de intensidad en la dirección horizontal (bordes verticales).

# ### 2. **Detección de bordes horizontales** (Kernel Sobel en dirección Y):
#    ```python
#    kernel_horizontal = np.array([[-1, -2, -1],
#                                  [0, 0, 0],
#                                  [1, 2, 1]])
#    ```
#    - Este kernel detecta cambios de intensidad en la dirección vertical, lo que significa que encuentra bordes horizontales.
#    - Los valores negativos en la parte superior y positivos en la parte inferior de la matriz indican que busca diferencias grandes de intensidad vertical.
#    - Esto hace que la salida de la operación resalte las áreas de la imagen donde los cambios de intensidad en la dirección vertical son más pronunciados (bordes horizontales).

# ### 3. **Combinación de bordes**:
#    Para combinar ambos resultados y obtener todos los bordes de la imagen, se puede calcular la magnitud del gradiente combinando las 
#    respuestas de los dos filtros (vertical y horizontal):
#    - La idea es que si un borde existe en cualquier dirección, ya sea vertical u horizontal, debería aparecer en la imagen combinada.
#    - Para esto, podemos usar la fórmula de la magnitud del gradiente:

#      
#      magnitud = sqrt{(G_x)^2 + (G_y)^2}

#      donde \( G_x \) es la respuesta del filtro Sobel en la dirección X (bordes verticales), y \( G_y \) es la respuesta en la dirección Y (bordes horizontales).

#    - Esta fórmula es similar a calcular la hipotenusa de un triángulo rectángulo y combina la intensidad de los bordes en ambas direcciones, dándonos una imagen 
#      que resalta todos los bordes.