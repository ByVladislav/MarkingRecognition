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