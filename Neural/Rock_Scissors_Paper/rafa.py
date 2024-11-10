import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DIR = 'Rock_Scissors_Paper/images/test'
TEST_DIR = 'Rock_Scissors_Paper/images/train'

train_dataset = image_dataset_from_directory(TRAIN_DIR, 
                                             shuffle=True,
                                             batch_size=32,
                                             image_size=(150, 150))

test_dataset = image_dataset_from_directory(TEST_DIR,
                                            shuffle=True,
                                            batch_size=32,
                                            image_size=(150, 150))

class_names = train_dataset.class_names
print(class_names)

def plot_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')


model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2), # Default 2
    
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax', name='output')
])

print(model.summary())

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model_history = model.fit(train_dataset, 
                          validation_data=test_dataset,
                          epochs=10)

test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Acc: {test_acc}')

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.plot(model_history.history['accuracy'], label='Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.legend(['Trian'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(model_history.history['loss'], label='Loss')
plt.plot(model_history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Trian'], loc='upper left')
plt.show()

model.save('rps_model.h5')