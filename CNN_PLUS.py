import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
import cv2  # Open cv

# train_dir = 'D:/Documents/Education/University/CHMNU/Основи Наукових Досліджень/programs/Topdata/train'
# test_dir = 'D:/Documents/Education/University/CHMNU/Основи Наукових Досліджень/programs/Topdata/test'
Datadirectory = "data/test/"  # Training data
Classes = ["0", "1", "2", "3", "4", "5", "6"]
img_size = 224


print("Підготовка ....... ")
training_Data = []


def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_Data()
print(len(training_Data))

import random

print("---- Змішую ----")
random.shuffle(training_Data)

X = []
Y = []

print("---- Визначаю марку ----")
for features, label in training_Data:
    X.append(features)
    Y.append(label)

print("---- Змінюю розмір ----")
X = np.array(X, dtype='uint8').reshape(-1, img_size, img_size, 3)
Y = np.array(Y, dtype='uint8')

print("---- Нормалізую ----")
X = X / 255.0

# Створення моделі
model = tf.keras.applications.MobileNetV2()  # This is the pre-trained model

base_input = model.layers[0].input
base_output = model.layers[-2].output

final_out = layers.Dense(128)(base_output)
final_out = layers.Activation('relu')(final_out)
final_out = layers.Dense(64)(final_out)
final_out = layers.Activation('relu')(final_out)
final_out = layers.Dense(len(Classes), activation='softmax')(final_out)  # adjusted for 7 Classes

# Компіляція моделі
new_model = keras.Model(inputs=base_input, outputs=final_out)
new_model.summary()
new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Навчання моделі
start_time = time.time()
new_model.fit(X, Y, epochs=30)
end_time = time.time()

# Виведення  значень

training_time = round(end_time - start_time, 3)

print(f"Час тренування: {training_time} секунд")

# Збереження моделі
new_model.save('test.h5')
