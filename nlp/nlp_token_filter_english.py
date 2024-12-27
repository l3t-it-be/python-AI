import spacy

# Загрузка словаря:
# python -m spacy download en_core_web_sm

# Загрузка английской модели
nlp = spacy.load('en_core_web_sm')

text = 'Apple is looking at buying a startup in San Francisco for $1 billion in 2024.'
doc = nlp(text)

filtered_tokens = [token.text for token in doc if not token.is_stop]
print('Токены без стоп-слов:', filtered_tokens)