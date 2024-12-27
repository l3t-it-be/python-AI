import spacy

# Загрузка словаря:
# python -m spacy download en_core_web_sm

# # Загрузка модели английского языка
nlp = spacy.load('en_core_web_sm')

# Обработка текста
text = 'Apple is looking at buying a startup in San Francisco for $1 billion in 2024.'
doc = nlp(text)

# Извлечение токенов
tokens = [token.text for token in doc]
print('Токены:', tokens)

# Извлечение частей речи
pos_tags = [(token.text, token.pos_) for token in doc]
print('Часть речи:', pos_tags)
