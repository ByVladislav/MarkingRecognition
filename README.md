# 🚀 Распознавание маркировки на трубах

Алгоритм, позволяет найти маркировку на трубе, распознать текст на ней и скорректировать его под маску

---

# 📌 Оглавление
* [Описание](#info)
* [Использование](#use)
* [Пример](#example)
* [Установка](#install)
* [Настройки](#settings)
* [Ошибки](#error)

---

<a id="info"></a>
# 📝 Описание
Алгоритм, выполняет обработку изобраэение в 5 этапов:
* По процентная обрезка, сверху и снизу
* Поиск рамки с помощью YOLO
* Распознование текста с помощью 0: base_ru 1: easy_ocr
* Структурирование текста для получения корректного результата
* Поиск схожих маркировок в списке и вывод предположений какой она может являтся

---

<a id="use"></a>
# 🛠 Использование

Главная функция выводит:
1. status - Обозначает возникли ли какие-нибудь ошибки во время распознования
2. mess - Выводит результат поиска маркировки или сообщения об ошибки
3. text - Текст после обработки OCR
4. marker - Итоговый результат
5. timer - Время выполнения в секундах
6. previously - Изображение после по процентной обрезки
7. markers - Список маркировок упорядоченый по схожестью с распознаным текстом
8. logs - Отображает ```dict()```, если отсутствуют изменения в тексте, или выводит то для чего была применена замена или корректировка Adding: было добавлено, Correction: Востановлено, например ```{'error_code': '003', 'error': ['Adding ТМК', 'Correction factory: ТПЗ->ЧТПЗ', 'Correction steel: ЗОГ2->30Г2']}```
9. adjustments - Выводит сообщение о том что не удалось точно распознать ноиер маркировки, и необходимо самостоятельно по представленому списку выбрать ```{"error_code": "004", "error": "Unable to accurately determine the room number"}```

---

Библиотека принимает на вход:
* path - Путь до файла *.json с настройками

```
record = Record() # Установление базового файла files/settings.json
record = Record(r"mysettings.json") # Установка своих настроек
```

---

Главная функция принимает:
* input - Входное изображение
* numModel - Номер модели 0: base_ru 1: easy_ocr
* pipe_numbers - Список возможных номеров труб (опционно)
* structure - Описание того что должно присутствовать на маркировке, пример {"company": "ТМК", "factory": "ЧТПЗ", "steel": "30Г2"}

```
status, mess, text, marker, timer, previously, markers, logs, adjustments = record(img, 1)
status, mess, text, marker, timer, previously, markers, logs, adjustments = record(img, 0, ['01693', '01699', '02691', '51691', '52691'])
status, mess, text, marker, timer, previously, markers, logs, adjustments = record(img, 1, ['01693', '01699', '02691', '51691', '52691'], {"company": "ТМК", "factory": "ЧТПЗ", "steel": "30Г2", "number": 5})
```

---

<a id="example"></a>
# ✨ Пример

Пример использование библиотеки находится в main.py, пример обработки потокового видео в camera.py

``` main.py
from Recognition import Record
import cv2

# Инициализируем декодер
record = Record()

# Открываем изображение
img = cv2.imread("img.jpg")

# Обрабатываем маркировку
status, mess, text, marker, timer, previously, markers, logs, adjustments = record(img, 1, ['01693', '01699', '02691', '51691', '52691'])

# Подготовленое изображение
cv2.imwrite("1.jpg", previously)

# Выводим результат
if status:
    print('--------------------------')

    # Результат работы
    print(text, " to ", marker)
    print('--------------------------')

    # Время выполнения
    print(timer, " sec.")
    print('--------------------------')

    # Отображение логов программы
    if logs != {} or adjustments != None:
        if logs != {}: print("Notes: ", logs)
        if adjustments != None: print(adjustments)
        print('--------------------------')

    # Вывод списка возможных вариантов
    if len(markers) > 1:
        print("List of options")
        for i in markers: print(i)
        print('--------------------------')

    # Итоговое изображение
    cv2.imwrite("2.jpg", mess)
else:
    print(mess)
```

---

<a id="install"></a>
# 📲 Установка

Чтобы запустить проект, выполните:

```bash
git clone https://github.com/ByVladislav/MarkingRecognition.git
cd MarkingRecognition
pip install -r requirements.txt
```

---

Установить trocr-base-ru, по путю MarkingRecognition/files/trocr-base-ru

Ссылка на репозиторий: https://huggingface.co/raxtemur/trocr-base-ru/tree/main

---

В файле **MarkingRecognition\\.venv\Lib\site-packages\easyocr\utils.py** в строчке 582 исправить ```maximum_y,maximum_x = img.shape``` на ```maximum_y,maximum_x, _ = img.shape```

---

<a id="settings"></a>
# ⚙️ Настройки

В настройку входит 7 параметров:
1. top_percent - Процент обрезки сверхк
2. bottom_percent - Процент обрезки снизу
3. height - Высота кадра после фильтрации (уменьшение размера сглаживает разрывы между линиями букв)
4. Significance - Порог совпадение для включения в спискок похожих маркировок
5. BestScore - Порог фильтрации идеального (конечного) результата
6. path_YOLO - Путь до YOLO модели
7. path_TROCR - Путь до модели распознования текста TrOCR

---

Пример файла настроек:
```
{
  "top_percent": 0.45,
  "bottom_percent": 0.35,
  "height": 75,
  "Significance": 0.5,
  "BestScore": 0.6,
  "path_YOLO": "files/model v1.pt",
  "path_TROCR": "files/trocr-base-ru"
}
```

---

<a id="error"></a>
# 📎 Обработка ошибок

* ```{"error_code":"001", "error":"Not detecting marking"}``` - Обозначение того что модель YOLO не нашла рамку
* ```{"error_code":"002", "error":"Unable to find pipe number"}``` - Обозначает то что при структурировании текста не удалось найти номер трубы
* ```{'error_code': '003', 'error': ['Adding ТМК', 'Correction factory: ТПЗ->ЧТПЗ]}``` - Обозначает вывод логов при структурировании текста
* ```{"error_code": "004", "error": "Unable to accurately determine the room number"}``` - Обозначает сообщение о том что не удалось точно распознать ноиер маркировки, и необходимо самостоятельно по представленому списку выбрать

---
