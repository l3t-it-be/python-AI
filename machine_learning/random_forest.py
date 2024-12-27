from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Данные: [Возраст, Доход (в тысячах долларов)]
# Данные представляют возраст и доход людей, а y — купил ли продукт (1 — да, 0 — нет)
X = [
    [25, 50],
    [30, 60],
    [35, 70],
    [40, 80],
    [45, 90],
    [50, 100],
    [55, 110],
    [60, 120],
]
y = [0, 0, 0, 0, 1, 1, 1, 1]  # Метки классов: 1 — купил, 0 — не купил

# Разделение на обучающую и тестовую выборки (70% обучение, 30% тест)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Создание и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}\n')

# Выводим предсказания с признаками
print('Возраст | Доход | Купил (Истинное) | Купил (Предсказание)')
print('---------------------------------------------------------')
for i in range(len(X_test)):
    age, income = X_test[i]
    actual = 'Да' if y_test[i] == 1 else 'Нет'
    predicted = 'Да' if y_pred[i] == 1 else 'Нет'
    print(f'{age:6} | {income:5} | {actual:^15} | {predicted:^18}')
