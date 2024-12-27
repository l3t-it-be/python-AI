import cv2

# Чтение изображения
image = cv2.imread('../images/python.png')

# Получение размеров изображения
height, width, channels = image.shape
print(f'Размер изображения: {width}x{height}')
print(f'Количество каналов: {channels}')
