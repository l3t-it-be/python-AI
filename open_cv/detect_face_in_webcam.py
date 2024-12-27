import cv2

# Загрузка каскада Хаара для распознавания лиц
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Захват видеопотока с веб-камеры (устройство с индексом 0)
video_capture = cv2.VideoCapture(0)

while True:
    # Захват кадра из видеопотока
    ret, frame = video_capture.read()

    # Преобразование кадра в оттенки серого для улучшения производительности
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на текущем кадре
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Рисование прямоугольников вокруг обнаруженных лиц
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение текущего кадра с обнаруженными лицами
    cv2.imshow('Video - Face Detection', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение захвата видеопотока и закрытие всех окон
video_capture.release()
cv2.destroyAllWindows()
