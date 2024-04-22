import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

train_dir = 'D:/Documents/Education/University/CHMNU/Основи Наукових Досліджень/programs/FER2013/train'
test_dir = 'D:/Documents/Education/University/CHMNU/Основи Наукових Досліджень/programs/FER2013/test'
img_size = 48

# Генераторів даних
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(img_size, img_size),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset=None
)


validation_generator = validation_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_size, img_size),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset=None
)

# Класи
class_names = list(train_generator.class_indices.keys())
print("Клас:", class_names)

# Створення моделі
model = Sequential()

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(len(class_names), activation='softmax'))


# Компіляція моделі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback для перевірки EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Навчання моделі
start_time = time.time()  # Запам'ятовуємо час початку навчання
history = model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[early_stopping])
end_time = time.time()  # Запам'ятовуємо час завершення навчання

# Виведення  значень
final_accuracy = model.evaluate(validation_generator)[1]
training_time = round(end_time - start_time, 3)

print(f"\nФінальна точність: {final_accuracy}")
print(f"Час Навчання: {training_time} секунд")

# Збереження моделі
model.save('basic_CNN_30.h5')