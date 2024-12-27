from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = 'Python is an amazing programming language with a lot of features. I love coding in Python!'

# Создание облака слов
wordcloud = WordCloud(
    width=800, height=400, background_color='white'
).generate(text)

# Отображение облака слов
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('../images/wordcloud.png')
plt.show()
