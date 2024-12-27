import cv2
import pytesseract

# Путь к tesseract: r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Загрузка каскада Хаара для распознавания номерных знаков
plate_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
)

# Загрузка изображения
image_path = '../images/car.png'
image = cv2.imread(image_path)

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение номерных знаков на изображении
plates = plate_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

# Проход по всем обнаруженным номерным знакам
for i, (x, y, w, h) in enumerate(plates):
    # Рисование прямоугольников вокруг обнаруженных номерных знаков
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Вырезка области номерного знака для распознавания текста
    plate_region = gray_image[y : y + h, x : x + w]

    # Сохранение вырезанной области номерного знака в файл
    plate_region_path = f'../images/car_number.png'
    cv2.imwrite(plate_region_path, plate_region)
    print(f'Сохранено изображение номерного знака: {plate_region_path}')

    # Использование Tesseract для распознавания текста
    plate_text = pytesseract.image_to_string(plate_region, config='--psm 8')
    print(f'Распознанный текст номерного знака: {plate_text}')

    # Отображение вырезанной области номерного знака
    cv2.imshow('Plate Region', plate_region)

# Отображение изображения с обнаруженными номерными знаками
cv2.imshow('Car with Detected Plates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
