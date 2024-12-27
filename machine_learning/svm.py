import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Пример данных (две группы, каждая с двумя признаками)
# Класс 0
X0 = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
# Класс 1
X1 = np.array([[6, 5], [7, 8], [8, 8], [7, 5], [8, 6]])

# Объединяем данные и создаем метки
X = np.vstack((X0, X1))
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Метки классов (0 и 1)

# Создание и обучение модели SVM с линейным ядром
model = SVC(kernel='linear')
model.fit(X, y)

# Предсказание меток классов на тех же данных (для простоты)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f'Точность модели SVM: {accuracy:.2f}')

# Визуализация данных и разделяющей гиперплоскости
plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Класс 0')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Класс 1')

# Отображение разделяющей линии
w = model.coef_[0]
b = model.intercept_[0]
x_points = np.linspace(0, 10, 10)
y_points = -(w[0] / w[1]) * x_points - b / w[1]
plt.plot(
    x_points,
    y_points,
    color='green',
    linestyle='--',
    label='Разделяющая линия',
)

plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

plt.title('Пример классификации SVM с линейным ядром')
plt.savefig('../images/svm_classification.png')
plt.show()
