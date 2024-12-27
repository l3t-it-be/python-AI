from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Пример данных о клиентах: [Возраст, Доход (в тысячах долларов)]
X = np.array(
    [
        [25, 30],
        [30, 35],
        [35, 45],
        [40, 50],
        [45, 60],
        [50, 65],
        [55, 70],
        [60, 80],
        [65, 85],
        [70, 90],
        [20, 20],
        [25, 25],
        [30, 30],
        [35, 35],
        [40, 40],
        [60, 60],
        [65, 65],
        [70, 70],
        [75, 75],
        [80, 80],
    ]
)

# Создание и обучение модели K-средних с числом кластеров 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Предсказание кластеров для данных
labels = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Визуализация результатов
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='viridis',
    marker='o',
    edgecolor='k',
    label='Клиенты',
)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='red',
    s=200,
    alpha=0.75,
    label='Центры кластеров',
)
plt.xlabel('Возраст')
plt.ylabel('Доход (в тыс. $)')
plt.legend()

plt.title('Сегментация клиентов методом K-средних')
plt.savefig('../images/clients_segmentation.png')
plt.show()
