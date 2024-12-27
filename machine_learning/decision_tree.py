from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Данные: Признаки - [Температура, Влажность, Ветер (1 - дует, 0 - нет)]
X = [
    [22, 65, 0],
    [23, 70, 1],
    [24, 85, 1],
    [25, 95, 0],
    [20, 80, 0],
    [27, 72, 1],
    [21, 75, 0],
    [30, 60, 1],
]
y = [1, 0, 0, 0, 1, 1, 1, 1]  # Целевая переменная: 1 - играет, 0 - не играет

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Создание и обучение модели
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')

# Предсказание для нового примера
new_example = [[26, 70, 1]]  # Температура = 26, Влажность = 70, Ветер - дует
play_prediction = model.predict(new_example)
print(
    'Будет ли человек играть в теннис?',
    'Да' if play_prediction[0] == 1 else 'Нет',
)
