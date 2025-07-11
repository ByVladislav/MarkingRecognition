from Recognition import Record
import cv2

# Инициализируем декодер
record = Record()

# Открываем изображение
img = cv2.imread("img.jpg")

# Обрабатываем маркировку
status, mess, text, marker, timer = record(img, 0)

# Выводим результат
if status:
    print(marker)
    print(timer, " сек.")
else: print(mess)