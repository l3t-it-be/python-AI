import cv2
from PIL import Image, ImageDraw


def make_mask(width, height):
    # Создаем новое изображение с RGBA (R, G, B, A) и полностью прозрачным фоном
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Рисуем красный круг на изображении
    radius = width // 8  # Уменьшаем радиус круга
    center_x = width // 2
    center_y = height // 2
    draw.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        fill=(255, 0, 0, 255),
    )

    # Сохраняем изображение
    image.save('../images/mask_with_alpha.png', 'PNG')


make_mask(200, 200)

# Загрузка каскада Хаара для распознавания лиц
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Загрузка изображения маски
mask_image = cv2.imread('../images/mask_with_alpha.png', cv2.IMREAD_UNCHANGED)

# Проверка, имеет ли маска альфа-канал (четыре канала)
has_alpha = mask_image.shape[2] == 4

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

    # Наложение маски на каждое обнаруженное лицо
    for x, y, w, h in faces:
        # Изменение размера маски в соответствии с размером обнаруженного лица
        mask_resized = cv2.resize(mask_image, (w, h))

        if has_alpha:
            # Если у маски есть альфа-канал
            for ch in range(h):
                for p in range(w):
                    # Если пиксель маски не прозрачный (альфа-канал не 0), заменяем пиксель на кадре
                    if (
                        mask_resized[ch, p][3] != 0
                    ):  # Проверяем альфа-канал маски
                        frame[y + ch, x + p] = mask_resized[ch, p][
                            :3
                        ]  # Копируем только BGR-каналы
        else:
            # Если у маски нет альфа-канала, просто наложим маску
            frame[y : y + h, x : x + w] = mask_resized

    # Отображение текущего кадра с наложенной маской
    cv2.imshow('Video - Face Mask', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение захвата видеопотока и закрытие всех окон
video_capture.release()
cv2.destroyAllWindows()
