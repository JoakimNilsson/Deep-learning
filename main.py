from time import sleep
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

age_classifier = load_model(r'C:\Users\jocke\Desktop\Skola\Deep learning\end_task\my_age_model_1.h5')
emotion_classifier = load_model(r'C:\Users\jocke\Desktop\Skola\Deep learning\end_task\my_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_classifier = cv2.CascadeClassifier(r'C:\Users\jocke\Desktop\Skola\Deep learning\end_task\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        age_input = cv2.resize(roi_gray, (48, 48))
    
        age_input = cv2.cvtColor(age_input, cv2.COLOR_GRAY2BGR)

        age_input = age_input.reshape(-1, 48, 48, 3) 
        age_input = age_input.astype('float32') 
        age_input /= 255.0

        age_prediction = age_classifier.predict(age_input)
        predicted_age_group = "Unknown"

        age_groups = ["20-30", "31-40", "41-50"]

        age_group_index = np.argmax(age_prediction)

        if 0 <= age_group_index < len(age_groups):
            predicted_age_group = age_groups[age_group_index]

        roi = roi_gray.astype('float')
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        emotion_prediction = emotion_classifier.predict(roi)[0]
        predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        text = f'Age Group: {predicted_age_group}, Emotion: {predicted_emotion}'
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Age and Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()