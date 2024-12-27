import spacy

# Загрузка словаря:
# python -m spacy download ru_core_news_sm

# Загрузка русской модели
nlp = spacy.load('ru_core_news_sm')

text = 'Компания Apple рассматривает возможность покупки стартапа в Сан-Франциско за 1 миллиард долларов в 2024 году.'
doc = nlp(text)

# Удаление стоп-слов
filtered_tokens = [token.text for token in doc if not token.is_stop]
print('Токены без стоп-слов:', filtered_tokens)
