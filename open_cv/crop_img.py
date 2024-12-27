import cv2

# Чтение изображения
image = cv2.imread('../images/python.png')

# Получение размеров изображения
height, width = image.shape[:2]

# Задание размеров области для обрезки
crop_width = 700  # Ширина области для обрезки
crop_height = 700  # Высота области для обрезки

# Вычисление координат центральной области
x_start = (width - crop_width) // 2
y_start = (height - crop_height) // 2
x_end = x_start + crop_width
y_end = y_start + crop_height

# Обрезка изображения
cropped_image = image[y_start:y_end, x_start:x_end]

# Сохранение обрезанного изображения
cv2.imwrite('../images/cropped_python.png', cropped_image)

# Отображение обрезанного изображения
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
