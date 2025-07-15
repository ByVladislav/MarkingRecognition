from Recognition import Record
import cv2

# Инициализируем декодер
record = Record(0.45, 0.35)

# Открываем изображение
img = cv2.imread("img.jpg")

# Обрабатываем маркировку
status, mess, _, marker, timer, _ = record(img, 0)

# Выводим результат
if status:
    print(marker)
    print(timer, " сек.")
else: print(mess)