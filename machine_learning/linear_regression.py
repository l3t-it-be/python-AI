from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Данные, соответствующие зависимости y = 2x + 1
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [2 * x[0] + 1 for x in X]  # Генерация y на основе формулы y = 2x + 1

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Выводим предсказанные и истинные значения
for i in range(len(X_test)):
    print(
        f'Для X = {X_test[i][0]}, истинное значение y = {y_test[i]}, предсказанное значение y = {y_pred[i]:.1f}'
    )
