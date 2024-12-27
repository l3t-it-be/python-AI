import tensorflow as tf
import numpy as np

# Генерация данных для линейной регрессии
x_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y_train = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

# Создание модели
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(1,)),  # Использование Input(shape)
        tf.keras.layers.Dense(units=1),  # Один нейрон, один входной параметр
    ]
)

# Компиляция модели
model.compile(optimizer='sgd', loss='mean_squared_error')

# Обучение модели
model.fit(x_train, y_train, epochs=500, verbose=1)

# Проверка предсказания
print(model.predict(np.array([5.0], dtype=np.float32)))
