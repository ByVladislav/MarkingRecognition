from Recognition import Record
import cv2

# Инициализируем декодер
record = Record()


# Открываем изображение
img = cv2.imread("img.jpg")

# Обрабатываем маркировку
_, text, marker, timer = record(img)

# Выводим результат
print(marker)
print(timer, " сек.")