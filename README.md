# 🚀 Распознавание маркировки на трубах

Алгоритм, позволяет найти маркировку на трубе, распознать текст на ней и скорректировать его под маску

---

# 📌 Оглавление
* [Описание](#info)
* [Использование](#use)
* [Пример](#example)
* [Установка](#install)
* [Ошибки](#error)

---

<a id="info"></a>
# 📝 Описание
Алгоритм, выполняет обработку изобраэение в 4 этаппа:
* По процентная обрезка, сверху и снизу
* Поиск рамки с помощью YOLO
* Распознование текста с помощью 0: base_ru 1: easy_ocr
* Структурирование текста для получения корректного результата

---

<a id="use"></a>
# 🛠 Использование

Главная функция выводит:
1. status - Обозначает возникли ли какие-нибудь ошибки во время распознования
2. mess - Выводит результат поиска маркировки или сообщения об ошибки
3. text - Текст после обработки OCR
4. marker - Итоговый результат
5. timer - Время выполнения в секундах
6. color - Изображение после по процентной обрезки
7. corrected - Отображает ```dict()```, если отсутствуют изменения в тексте, или выводит то для чего была применена замена или корректировка Adding: было добавлено, Correction: Востановлено, например ```{'error_code': '003', 'error': ['Adding ТМК', 'Correction ЧТПЗ']}```

---

Библиотека принимает на вход:
* top - Процент обрезки по вверху 0-1
* bottom - Процент обрезки по низу 0-1
* pathModel - Путь до YOLO модели, изначально files/model v2.pt
* pathOCR - Путь до base_ocr, изначально files/trocr-base-ru

```
record = Record(0.45, 0.35) # Сокращёное применение
record = Record(0.45, 0.35, r"files/model v2.pt", r"files/trocr-base-ru") # Полное применение
```

---

Главная функция принимает:
* input - Входное изображение
* numModel - Номер модели 0: base_ru 1: easy_ocr 
* structure - Описание того что должно присутствовать на маркировке, пример {"company": "ТМК", "factory": "ЧТПЗ", "steel": "30Г2"}

```status, mess, text, marker, timer, pred, corrected = record(frame, 0)```

---

<a id="example"></a>
# ✨ Пример

Пример использование библиотеки находится в main.py, пример обработки потокового видео в camera.py

``` main.py
from Recognition import Record
import cv2

# Инициализируем декодер
record = Record(0.45, 0.35)

# Открываем изображение
img = cv2.imread("img.jpg")

# Обрабатываем маркировку
status, mess, _, marker, timer, _, _ = record(img, 0)

# Выводим результат
if status:
    print(marker)
    print(timer, " сек.")
else: print(mess)
```

---

<a id="install"></a>
# ⚙️ Установка

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

В файле **MarkingRecognition\.venv\Lib\site-packages\easyocr\utils.py** в строчке 582 исправить ```maximum_y,maximum_x = img.shape``` на ```maximum_y,maximum_x, _ = img.shape```

---

<a id="error"></a>
# 📎 Обработка ошибок

* ```{"error_code":"001", "error":"Not detecting marking"}``` - Обозначение того что модель YOLO не нашла рамку
* ```{"error_code":"002", "error":"Unable to find pipe number"}``` - Обозначает то что при структурировании текста не удалось найти номер трубы
* ```{'error_code': '003', 'error': ['Adding ТМК', 'Correction ЧТПЗ']}``` - Обозначает вывод логов при структурировании текста

---
