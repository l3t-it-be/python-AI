import cv2

# Чтение изображения
image = cv2.imread('../images/python.png')

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Сохранение серого изображения
cv2.imwrite('../images/gray_python.png', gray_image)

# Отображение изображений
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
