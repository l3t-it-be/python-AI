import cv2

# Чтение изображения
image = cv2.imread('../images/python.png')

# Получение размеров изображения
(h, w) = image.shape[:2]

# Определение центра изображения
center = (w // 2, h // 2)

# Угол поворота
angle = 45

# Получение матрицы поворота
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

# Применение матрицы поворота
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# Отображение повернутого изображения
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
