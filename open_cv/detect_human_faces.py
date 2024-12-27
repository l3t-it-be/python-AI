import cv2

# Загрузка каскада Хаара для распознавания лиц
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Загрузка изображения
image_path = '../images/smiling_people.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

# Рисование прямоугольников вокруг обнаруженных лиц
for x, y, w, h in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Сохранение изображения с обнаруженными лицами
output_path = '../images/detected_faces.jpg'
cv2.imwrite(output_path, image)

# Отображение изображения с обнаруженными лицами
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
