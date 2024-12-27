import cv2

# Чтение изображения
image = cv2.imread('../images/python.png')

brighter_image = cv2.convertScaleAbs(image, alpha=1.0, beta=50)

# Увеличение контрастности (умножение на 1.5)
higher_contrast_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

# Отображение изображений с измененной яркостью и контрастностью
cv2.imshow('Brighter Image', brighter_image)
cv2.imshow('Higher Contrast Image', higher_contrast_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
