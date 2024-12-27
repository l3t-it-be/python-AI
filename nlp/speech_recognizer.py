import speech_recognition as sr

recognizer = sr.Recognizer()


with sr.Microphone() as source:
    print('Скажи что-нибудь...')
    # Запись звука с микрофона
    audio_data = recognizer.listen(source)
    print('Запись завершена. Распознавание...')

    try:
        # Распознавание речи с использованием Google Web Speech API
        text = recognizer.recognize_google(
            audio_data, language='ru-RU'
        )  # Параметр 'ru-RU' для русского языка
        print('Распознанный текст:', text)
    except sr.UnknownValueError:
        print('Речь не распознана')
    except sr.RequestError as e:
        print(f'Ошибка сервиса; {e}')
