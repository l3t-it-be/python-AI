from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Данные: возраст и наличие болезни (1 — есть, 0 — нет)
X = [[22], [25], [47], [52], [46], [56], [55], [60], [62], [63]]
y = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # Целевая переменная: 0 — здоров, 1 — болен

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Создание и обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print('Точность модели:', accuracy)

# Вывод вероятностей принадлежности к классу "1" (болен) для тестовых данных
y_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для класса "1"
for i in range(len(X_test)):
    print(
        f'Возраст: {X_test[i][0]}, вероятность заболевания: {y_proba[i]:.2f}'
    )
