import cv2

# Загрузка каскада Хаара для распознавания кошачьих мордочек
cat_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
)

# Загрузка изображения
image_path = '../images/cats.jpg'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого для улучшения производительности
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение кошачьих мордочек на изображении
cats = cat_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10)
)

# Рисование прямоугольников вокруг обнаруженных кошачьих мордочек
for x, y, w, h in cats:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Сохранение изображения с обнаруженными кошачьими мордочками
output_path = '../images/detected_cats.jpg'
cv2.imwrite(output_path, image)

# Отображение изображения с обнаруженными кошачьими мордочками
cv2.imshow('Cat Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
