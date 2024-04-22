import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('Done_Models/model_30_CNNPlus.h5')


def predict_emotion(image_path):
    img = cv2.imread(image_path)
    img_copy = img.copy()

    # Виділення обличчя OpenCV (хаар каскад)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Перевірка
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = img[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = np.expand_dims(face_roi, axis=0) / 255.0

            # Передбачення
            predictions = model.predict(face_roi)

            # Отримання емоції та точності для обличчя
            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_index = np.argmax(predictions)
            predicted_emotion = emotions[emotion_index]
            confidence = predictions[0, emotion_index] * 100

            print(f'Емоція: {predicted_emotion}, Точність: {confidence:.2f}%')

            cv2.putText(img_copy, f'Emotion: {predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            cv2.putText(img_copy, f'Confidence: {confidence:.2f}%', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

    # Візуалізація
    cv2.imshow('Emotion Recognition', img_copy)
    cv2.waitKey(0)

image_path = 'happy.jpg'
predict_emotion(image_path)
cv2.destroyAllWindows()
